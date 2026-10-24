#!/usr/bin/env bash
# Copy to scripts/env.local.sh and configure the REMOTE Linux paths.
# Do not put secret keys in this file. Export keys in the terminal.
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export DATA_ROOT="${DATA_ROOT:-$HOME/datasets/MMTab-eval}"
export MMTAB_ANNOTATIONS="${MMTAB_ANNOTATIONS:-$DATA_ROOT/MMTab-eval_test_data_49K.json}"
export MMTAB_IMAGES="${MMTAB_IMAGES:-$DATA_ROOT/MMTab-eval_table_images_23K}"
export MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
# Optional: exact local checkpoint folder instead of downloading a model ID.
# export MODEL_WEIGHTS="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
# export MODEL_REVISION="an immutable upstream checkpoint commit"
export PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export API_PROXY_MODE="${API_PROXY_MODE:-direct}"
# AGENT_CIRCLE_RUNS and MANIFEST can point to scratch storage on the remote host.
# export AGENT_CIRCLE_RUNS="$HOME/experiments/agent-circle"
# export MANIFEST="$HOME/experiments/mmtab_eval.json"
