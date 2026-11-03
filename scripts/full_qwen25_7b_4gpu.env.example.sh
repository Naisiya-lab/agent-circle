#!/usr/bin/env bash
# Defaults match the remote paths supplied by the user.
# Optional overrides: copy to full_qwen25_7b_4gpu.env.local.sh; never put keys here.
export PYTHON_BIN="${PYTHON_BIN:-python}"
export SOURCE_MANIFEST="${SOURCE_MANIFEST:-/nc1test1/lwj/data/MMTab_eval/prepared/mmtab_eval.json}"
export MODEL_WEIGHTS="${MODEL_WEIGHTS:-/nc1test1/lwj/Models/Qwen2.5-VL-7B-Instruct}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-vl-7b}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export FIRST_PORT="${FIRST_PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_7b_full4gpu_v2}"
export AGENT_CIRCLE_RUNS="${AGENT_CIRCLE_RUNS:-$PROJECT_ROOT/runs}"
export SHARD_ROOT="${SHARD_ROOT:-/nc1test1/lwj/data/MMTab_eval/prepared/$EXPERIMENT_NAME}"
export CONTROL_ROOT="${CONTROL_ROOT:-$AGENT_CIRCLE_RUNS/$EXPERIMENT_NAME}"
