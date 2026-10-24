#!/usr/bin/env bash
# Official nominal size: 7B. Set MODEL_REVISION to an immutable checkpoint commit for experiments.
export MODEL_ID='llava-hf/llava-1.5-7b-hf'
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:${PORT:-8000}/v1}"
