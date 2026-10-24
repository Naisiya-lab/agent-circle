#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
PROFILE="${1:-qwen25_vl_7b}"
[[ "$PROFILE" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'Invalid profile'; exit 2; }
source "$PROJECT_ROOT/scripts/models/$PROFILE.sh"
WEIGHTS="${MODEL_WEIGHTS:-$MODEL_ID}"
EXTRA_ARGS=()
if [[ -n "${MODEL_REVISION:-}" ]]; then EXTRA_ARGS+=(--revision "$MODEL_REVISION"); fi
# A server is shared by all four roles; do not load four copies for four agents.
# Four A800s can run four single-GPU replicas, on ports 8000..8003.
# These are conservative starting settings, not measured memory/throughput guarantees.
exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$WEIGHTS" --served-model-name "${SERVED_MODEL_NAME:-$MODEL_ID}" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --gpu-memory-utilization 0.85 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --generation-config vllm --trust-remote-code "${EXTRA_ARGS[@]}"
