#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# Default: preserve all official tasks, including T2T and OOD tasks.
# For a PREDECLARED subset: bash scripts/prepare_eval.sh --exclude-tasks T2T
"$PYTHON_BIN" agent_circle.py prepare-data --dataset "$MMTAB_ANNOTATIONS" \
  --images "$MMTAB_IMAGES" --output "$MANIFEST" "$@"
