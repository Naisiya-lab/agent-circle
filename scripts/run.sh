#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
PROFILE="${1:-qwen25_vl_7b}"; MODE="${2:-circle}"
[[ "$PROFILE" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'Invalid profile'; exit 2; }
[[ "$MODE" == baseline || "$MODE" == circle ]] || { echo 'mode must be baseline/circle'; exit 2; }
if (( $# >= 2 )); then shift 2; elif (( $# == 1 )); then shift; fi
source "$PROJECT_ROOT/scripts/models/$PROFILE.sh"
ROUTE="${ROUTE:-B}"
RUN_NAME="${RUN_NAME:-${PROFILE}_${MODE}_${ROUTE}_$(date +%Y%m%d)}"
exec "$PYTHON_BIN" agent_circle.py run --mode "$MODE" --route "$ROUTE" \
  --model-profile "$PROJECT_ROOT/configs/models/$PROFILE.json" --source "$MANIFEST" \
  --run "$RUN_NAME" "$@"
