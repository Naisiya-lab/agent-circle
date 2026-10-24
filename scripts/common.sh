#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/scripts/env.local.sh" ]]; then
  source "$PROJECT_ROOT/scripts/env.local.sh"
else
  source "$PROJECT_ROOT/scripts/env.example.sh"
fi
export MANIFEST="${MANIFEST:-$PROJECT_ROOT/prepared/mmtab_eval.json}"
export AGENT_CIRCLE_RUNS="${AGENT_CIRCLE_RUNS:-$PROJECT_ROOT/runs}"
cd "$PROJECT_ROOT"
