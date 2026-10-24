#!/usr/bin/env bash
# Official nominal size: 4B. Set MODEL_REVISION to an immutable checkpoint commit for experiments.
export MODEL_ID='OpenGVLab/InternVL2_5-4B'
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:${PORT:-8000}/v1}"
