#!/usr/bin/env bash
# Deployment and job scheduling only. Does not source the single-GPU env.local.sh.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/scripts/full_qwen25_7b_4gpu.env.local.sh" ]]; then
  source "$PROJECT_ROOT/scripts/full_qwen25_7b_4gpu.env.local.sh"
else
  source "$PROJECT_ROOT/scripts/full_qwen25_7b_4gpu.env.example.sh"
fi
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/full_qwen25_7b_4gpu.py" "${1:-run}" \
  --project "$PROJECT_ROOT" --source "$SOURCE_MANIFEST" --weights "$MODEL_WEIGHTS" \
  --shards-root "$SHARD_ROOT" --control-root "$CONTROL_ROOT" --runs-root "$AGENT_CIRCLE_RUNS" \
  --experiment "$EXPERIMENT_NAME" --model-name "$SERVED_MODEL_NAME" \
  --gpus "$GPU_IDS" --first-port "$FIRST_PORT" --context "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" --memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --start-timeout "$SERVER_START_TIMEOUT" "${@:2}"
