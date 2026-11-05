"""Persist per-sample, per-role execution budgets across rework and resume."""
from . import transport

ROLES = ("understand", "select", "solve", "check")


class Budgets:
    def __init__(self, folder, cfg):
        self.path = folder / "agent_budgets.json"
        self.limits = {"tool": cfg.get("agent_limits", {}).get("tool_calls", 5),
                       "image": cfg.get("agent_limits", {}).get("image_requests", 5)}
        for value in self.limits.values():
            transport.require(type(value) is int and value >= 0, "Agent limits must be nonnegative integers")
        self.state = transport.read_json(self.path) if self.path.exists() else {"operations": {}, "fallbacks": {}}

    def used(self, role, kind):
        return sum(x["role"] == role and x["kind"] == kind for x in self.state["operations"].values())

    def take(self, role, kind, key):
        key = role + ":" + kind + ":" + key
        if key in self.state["operations"]:
            return key  # Replay the same recorded operation, not a new allowance.
        if self.used(role, kind) >= self.limits[kind]:
            return None
        self.state["operations"][key] = {"role": role, "kind": kind, "time": transport.stamp()}
        self.save()
        return key

    def receipt(self, key):
        return self.state["operations"][key].get("response")

    def complete(self, key, response):
        self.state["operations"][key]["response"] = response
        self.save()

    def fallback(self, role, stage, reason):
        self.state["fallbacks"].setdefault(stage, {"role": role, "reason": reason, "time": transport.stamp()})
        self.save()

    def save(self):
        transport.save_json(self.path, self.state)

    def summary(self):
        return {"limits": self.limits,
                "by_role": {role: {kind: self.used(role, kind) for kind in self.limits} for role in ROLES},
                "fallbacks": self.state["fallbacks"]}
