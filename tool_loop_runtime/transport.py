"""Standalone API transport utilities with a client-local proxy policy."""

from __future__ import annotations

import argparse

import base64

from collections import Counter, defaultdict, deque

from email.utils import parsedate_to_datetime

from functools import lru_cache

from datetime import datetime, timezone

import getpass

import hashlib

import html

import heapq

import http.client

import json

import mimetypes

import os

from pathlib import Path

import random

import re

import sys

import time

import urllib.error

import urllib.request

import uuid

BASE = Path(__file__).resolve().parent.parent

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def stamp():
    return datetime.now(timezone.utc).isoformat()

def log(path, message):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(stamp() + " " + message + "\n")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def text_api(cfg, role):
    require(role in ("select", "solve"), "No text API profile for " + role)
    name = "selector_api" if role == "select" else "solver_api"
    profile = cfg[name]
    require(isinstance(profile, dict), name + " must be an object")
    for key in ("endpoint", "key_env"):
        require(isinstance(profile.get(key), str) and bool(profile[key].strip()),
                name + " requires " + key)
    require(isinstance(profile.get("request_options"), dict), name + ".request_options must be an object")
    require(not ({"model", "messages", "stream", "max_tokens"} & profile["request_options"].keys()),
            name + ".request_options cannot override model/messages/stream/max_tokens")
    return profile


def uses_image(cfg, role):
    mode = cfg.get("selector_input", "text")
    require(mode in ("text", "image_and_html"), "selector_input must be text or image_and_html")
    return role in ("understand", "check") or (role == "select" and mode == "image_and_html")

class RunProgress:
    """Session attempts and actual saved results are different counters."""
    def __init__(self, pending, planned, completed):
        self.pending, self.planned, self.completed = pending, planned, completed
        self.finished, self.errors, self.width = 0, 0, 0
        self.started = time.monotonic()

    def show(self, sid="", stage="", phase=""):
        text = (f"[session {self.finished}/{self.pending}] results {self.completed}/{self.planned}"
                f" | errors {self.errors} | elapsed {(time.monotonic()-self.started)/60:.1f} min")
        if sid:
            text += f" | {sid} {stage} {phase}"
        self.width = max(self.width, len(text))
        print(chr(13)+text.ljust(self.width), end="", flush=True)

class ApiFailure(Exception):
    def __init__(self, message, fatal=False, retryable=False, http_status=None):
        super().__init__(message)
        self.fatal = fatal
        self.retryable = retryable
        self.http_status = http_status

def retry_after_seconds(value):
    """Honor the server's transport throttle, including HTTP-date headers."""
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0

def wait_with_progress(seconds, progress=None, sid="", stage="", phase="waiting"):
    """Interruptible countdown; no keys or sample outputs are printed."""
    until = time.monotonic() + max(0.0, seconds)
    remaining = max(0.0, seconds)
    while remaining > 0:
        if progress:
            progress(sid, stage, f"{phase} {remaining:.0f}s")
        time.sleep(min(1.0, remaining))
        remaining = max(0.0, until - time.monotonic())

def semantic_request_fingerprint(body):
    """Transport delivery mode does not change the task or model generation settings."""
    semantic = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
    return digest(json.dumps(semantic, ensure_ascii=False, sort_keys=True).encode("utf-8"))

class IncompleteStream(OSError):
    pass

def read_sse_response(response, event_path, on_event=None, deadline=None):
    """Read complete SSE events and reconstruct choices.message without mixing reasoning into content."""
    content, reasoning, event_count, finish = [], [], 0, None
    char_counts = {'content': 0, 'reasoning_content': 0}
    usage, metadata, data_lines, done = {}, {}, [], False
    event_path = Path(event_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as output:
        def event(data):
            nonlocal event_count, finish, done
            output.write(json.dumps({"received_at": stamp(), "data": data}, ensure_ascii=False) + "\n")
            output.flush()  # Keep already received fragments even after Ctrl+C or disconnect.
            event_count += 1
            if data.strip() == "[DONE]":
                done = True
                return
            obj = json.loads(data)
            require(isinstance(obj, dict), "Invalid SSE event")
            if obj.get("error"):
                raise ValueError("API stream error: " + json.dumps(obj["error"], ensure_ascii=False))
            for key in ("id", "model", "created", "request_id", "object"):
                if key in obj:
                    metadata[key] = obj[key]
            if isinstance(obj.get("usage"), dict) and obj["usage"]:
                usage.update(obj["usage"])
            for choice in obj.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                for key, target in (("content", content), ("reasoning_content", reasoning)):
                    value = delta.get(key)
                    if value is not None:
                        require(isinstance(value, str), "Non-text streaming " + key)
                        target.append(value)
                        char_counts[key] += len(value)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
            if on_event:
                on_event(event_count, char_counts["content"], char_counts["reasoning_content"])

        for raw_line in response:
            if deadline is not None and time.monotonic() > deadline:
                raise IncompleteStream("Streaming request exceeded total deadline")
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    event("\n".join(data_lines))
                    data_lines.clear()
                    if done:
                        break
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
            # Ignore SSE comments/keepalives and event/id/retry fields.
        if data_lines and not done:
            event("\n".join(data_lines))
    if finish is None:
        raise IncompleteStream("SSE ended before a finish_reason; partial events retained")
    raw = {**metadata, "choices": [{"index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": "".join(content),
                        "reasoning_content": "".join(reasoning)}}], "usage": usage,
           "_stream": {"events": event_count, "done_marker": done,
                       "event_file": str(event_path), "finish_received": True}}
    return raw

def proxy_mode(cfg):
    network = cfg.get("network", {})
    require(type(network) is dict, "network must be an object")
    mode = network.get("proxy_mode", "direct")
    require(mode in ("direct", "system"), "network.proxy_mode must be direct or system")
    return mode


def build_api_opener(cfg):
    # An explicit empty ProxyHandler also disables Windows registry proxy lookup.
    # Keep this local: do not install a global opener or modify os.environ.
    handler = urllib.request.ProxyHandler({}) if proxy_mode(cfg) == "direct" else urllib.request.ProxyHandler()
    return urllib.request.build_opener(handler)


def context_budget_adjustment(body, error_text):
    """Use the server's token counts; never truncate input messages/images."""
    maximum = re.search(r"maximum context length is\s+(\d+)\s+tokens", error_text, re.I)
    inputs = re.search(r"(\d+)\s+in (?:the )?(?:messages|input)", error_text, re.I)
    field = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
    requested = body.get(field)
    if not maximum or not inputs or type(requested) is not int:
        return None
    available = int(maximum.group(1)) - int(inputs.group(1))
    if available < 1 or available >= requested:
        return None
    return dict(body, **{field: available}), {
        "kind": "context_output_budget", "parameter": field, "previous": requested,
        "effective": available, "context_tokens": int(maximum.group(1)),
        "input_tokens": int(inputs.group(1)), "input_unchanged": True}


class ApiClient:
    def __init__(self, cfg, key, run_dir):
        self.cfg, self.key, self.run_dir = cfg, key, Path(run_dir)
        self.proxy_mode = proxy_mode(cfg)
        self.opener = build_api_opener(cfg)
        self.outage_started = None
        self.transient_streak = 0
        self.cooldowns = 0
        self.ready_at = 0.0

    def wait_for_transport(self, sid, stage):
        limit = self.cfg.get("outage_timeout_seconds", 900)
        delay = max(0.0, self.ready_at - time.monotonic())
        if self.outage_started is not None:
            remaining = limit - (time.monotonic() - self.outage_started)
            if remaining <= 0:
                raise ApiFailure(f"Communication has not recovered for {limit}s; saved stages retained.", fatal=True, retryable=True)
            delay = min(delay, remaining)
        wait_with_progress(delay, getattr(self, "progress", None), sid, stage, "network cooling")
        if self.outage_started is not None and time.monotonic() - self.outage_started >= limit:
            raise ApiFailure(f"Communication has not recovered for {limit}s; saved stages retained.", fatal=True, retryable=True)

    def call(self, body, sid, stage):
        retries = self.cfg.get("transport_retries", 4)
        body = dict(body)
        adjustments, failures = [], 0
        # One additional request may correct an explicitly reported context
        # overflow. Network retry accounting is independent of this correction.
        for attempt in range(retries + 2):
            self.wait_for_transport(sid, stage)
            started = time.monotonic()
            code, retry, fatal = None, False, False
            server_delay = 0.0
            try:
                if getattr(self, "progress", None):
                    self.progress(sid, stage, f"request attempt {attempt+1}")
                log(self.run_dir/"run.log", f"{sid} {stage} attempt={attempt+1} proxy_mode={self.proxy_mode} start")
                req = urllib.request.Request(self.cfg["endpoint"],
                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                             "Accept": "text/event-stream" if body.get("stream") else "application/json"},
                    method="POST")
                socket_timeout = self.cfg["timeout_seconds"]
                request_deadline = started + self.cfg.get("request_deadline_seconds", 1200)
                if self.outage_started is not None:
                    outage_deadline = self.outage_started + self.cfg.get("outage_timeout_seconds", 900)
                    socket_timeout = min(socket_timeout, max(0.1, outage_deadline-started))
                    request_deadline = min(request_deadline, outage_deadline)
                with self.opener.open(req, timeout=socket_timeout) as response:
                    headers_seconds = time.monotonic()-started
                    log(self.run_dir/"run.log", f"{sid} {stage} HTTP opened stream={body.get('stream', False)} headers_seconds={headers_seconds:.3f} proxy_mode={self.proxy_mode}")
                    content_type = getattr(response, "headers", {}).get("Content-Type", "")
                    if body.get("stream") and "application/json" not in content_type.lower():
                        last_log = [0.0]
                        def received(events, chars, reasoning_chars):
                            now = time.monotonic()
                            if events == 1 or now-last_log[0] >= 15:
                                last_log[0] = now
                                log(self.run_dir/"run.log",
                                    f"{sid} {stage} receiving events={events} content_chars={chars} reasoning_chars={reasoning_chars}")
                                if getattr(self, "progress", None):
                                    self.progress(sid, stage,
                                        f"receiving thinking_chars={reasoning_chars} output_chars={chars}")
                        event_path = self.run_dir/"samples"/sid/"streams"/(stage+"_"+uuid.uuid4().hex+".jsonl")
                        raw = read_sse_response(response, event_path, received, request_deadline)
                    else:
                        raw = json.load(response)
                require(isinstance(raw, dict) and raw.get("choices"), "API did not return choices")
                if self.outage_started is not None:
                    log(self.run_dir/"run.log", f"{sid} {stage} communication recovered")
                self.outage_started = None
                self.transient_streak = self.cooldowns = 0
                self.ready_at = 0.0
                raw["_transport"] = {"proxy_mode": self.proxy_mode, "successful_attempt": attempt+1,
                                     "response_headers_seconds": round(headers_seconds, 3),
                                     "successful_attempt_seconds": round(time.monotonic()-started, 3),
                                     "request_adjustments": adjustments,
                                     "effective_max_tokens": body.get("max_completion_tokens", body.get("max_tokens")),
                                     "effective_request_fingerprint": semantic_request_fingerprint(body)}
                return raw
            except urllib.error.HTTPError as exc:
                code = exc.code
                error = exc.read(1000).decode("utf-8", errors="replace").replace(self.key, "[REDACTED]")
                retry = code in (408, 429, 500, 502, 503, 504)
                fatal = code in (401, 402, 403, 404)
                if not retry:
                    # A concrete client-error response proves connectivity; do
                    # not carry an old network-outage timer into later samples.
                    self.outage_started = None
                    self.transient_streak = self.cooldowns = 0
                    self.ready_at = 0.0
                server_delay = retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
                message = f"HTTP {code}: {error}"
            except (TimeoutError, OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                message = f"{type(exc).__name__}: {exc}".replace(self.key, "[REDACTED]")
                retry = True
            except (ValueError, TypeError) as exc:
                # Malformed HTTP/SSE envelopes are transport failures, not judgments of agent output.
                message = f"{type(exc).__name__}: {exc}".replace(self.key, "[REDACTED]")
                retry = True
            event = {"time": stamp(), "id": sid, "stage": stage, "attempt": attempt+1,
                     "seconds": round(time.monotonic()-started, 3), "http_status": code, "error": message}
            with (self.run_dir / "api_attempts.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False)+"\n")
            log(self.run_dir/"run.log", f"{sid} {stage} {message}")
            correction = context_budget_adjustment(body, message) if code == 400 and not adjustments and self.cfg.get("context_budget_adaptation", True) else None
            if correction:
                body, receipt = correction
                adjustments.append(receipt)
                with (self.run_dir/"api_attempts.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps({"time": stamp(), "id": sid, "stage": stage, "request_adjustment": receipt})+"\n")
                log(self.run_dir/"run.log", f"{sid} {stage} adjusted output budget {receipt['previous']} -> {receipt['effective']}; input unchanged")
                continue
            if retry:
                now = time.monotonic()
                if self.outage_started is None:
                    self.outage_started = now
                self.transient_streak += 1
                base = self.cfg.get("retry_base_seconds", 5)
                cap = self.cfg.get("retry_max_seconds", 60)
                delay = min(cap, base * (2**failures)) * random.uniform(0.8, 1.2)
                if self.transient_streak >= self.cfg.get("cooldown_failure_threshold", 3):
                    cooling = min(self.cfg.get("cooldown_max_seconds", 120),
                                  self.cfg.get("cooldown_seconds", 30) * (2**self.cooldowns))
                    self.cooldowns += 1
                    self.transient_streak = 0
                    delay = max(delay, cooling)
                    log(self.run_dir/"run.log", f"{sid} {stage} circuit cooling {cooling:.1f}s; auto-resume enabled")
                delay = max(delay, server_delay)
                self.ready_at = max(self.ready_at, now + delay)
                log(self.run_dir/"run.log", f"{sid} {stage} retry wait {delay:.1f}s; attempt {attempt+1}/{retries+1}")
            if not retry or failures >= retries:
                raise ApiFailure(message, fatal=fatal, retryable=retry, http_status=code)
            failures += 1

def api_binding(cfg, role):
    if "role_apis" in cfg:
        profile = cfg["role_apis"][role]
        return profile["endpoint"], profile["key_env"]
    if role in ("select", "solve") and not uses_image(cfg, role):
        profile = text_api(cfg, role)
        return profile["endpoint"], profile["key_env"]
    return cfg["endpoint"], "ZHIPU_API_KEY"

def load_api_keys(cfg):
    """Ask once per endpoint/key binding; keys never enter config or run artifacts."""
    keys = {}
    for role in ("understand", "select", "solve", "check"):
        binding = api_binding(cfg, role)
        if binding in keys:
            continue
        endpoint, env_name = binding
        if "role_apis" in cfg and not cfg["role_apis"][role].get("require_key", True):
            keys[binding] = os.environ.get(env_name) or "EMPTY"
            continue
        aliases = (env_name, "GLM_API_KEY", "BIGMODEL_API_KEY") if env_name == "ZHIPU_API_KEY" else (env_name,)
        key = next((os.environ[name] for name in aliases if os.environ.get(name)), None)
        if not key:
            require(sys.stdin.isatty(), "No API key configured for " + env_name + ". Run interactively.")
            label = cfg["models"][role] if role in ("select", "solve") else "GLM"
            key = getpass.getpass(f"Paste {label} API key for {endpoint} (hidden; not saved): ").strip()
        require(bool(key), "Empty API key for " + env_name)
        keys[binding] = key
    return keys

class RoleApiClient:
    """Explicit per-role dispatch; providers may have independent endpoints and keys."""
    def __init__(self, cfg, keys, run_dir):
        self.cfg, self.keys, self.run_dir = cfg, keys, run_dir
        self.clients = {}
        self.progress = None

    def call(self, body, sid, stage, *, role):
        binding = api_binding(self.cfg, role)
        if binding not in self.clients:
            require(binding in self.keys, "No API key loaded for role " + role)
            self.clients[binding] = ApiClient(dict(self.cfg, endpoint=binding[0]),
                                              self.keys[binding], self.run_dir)
        client = self.clients[binding]
        client.progress = self.progress
        return client.call(body, sid, stage)

def saved_error_is_transient(path):
    """Resume communication errors automatically, including pre-upgrade error records."""
    error = read_json(path)
    if error.get("retryable") is True:
        return True
    if "retryable" in error or error.get("fatal"):
        return False
    return error.get("type") == "ApiFailure" and str(error.get("message", "")).startswith(
        ("URLError:", "TimeoutError:", "OSError:", "IncompleteStream:",
         "RemoteDisconnected:", "SSLError:", "HTTP 408:", "HTTP 429:",
         "HTTP 500:", "HTTP 502:", "HTTP 503:", "HTTP 504:"))
