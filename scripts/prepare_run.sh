#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
PROFILE="${1:-qwen25_vl_7b}"; MODE="${2:-circle}"
[[ "$PROFILE" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'Invalid profile'; exit 2; }
if (( $# >= 2 )); then shift 2; elif (( $# == 1 )); then shift; fi
source "$PROJECT_ROOT/scripts/models/$PROFILE.sh"
"$PYTHON_BIN" agent_circle.py prepare --mode "$MODE" --route "${ROUTE:-B}" \
 --model-profile "$PROJECT_ROOT/configs/models/$PROFILE.json" --source "$MANIFEST" \
 --run "${RUN_NAME:-${PROFILE}_${MODE}_${ROUTE:-B}_$(date +%Y%m%d)}" "$@"
