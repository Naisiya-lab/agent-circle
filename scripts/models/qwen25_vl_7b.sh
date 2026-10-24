#!/usr/bin/env bash
# Official nominal size: 7B. Set MODEL_REVISION to an immutable checkpoint commit for experiments.
export MODEL_ID='Qwen/Qwen2.5-VL-7B-Instruct'
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:${PORT:-8000}/v1}"
