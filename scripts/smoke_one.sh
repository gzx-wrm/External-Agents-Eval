#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
agent="${1:-pi}"
manifest="${2:-${repo_root}/manifests/lite_filtered_smoke1.json}"

exec "${repo_root}/scripts/run_with_deepseek_key.sh" \
  external-agents-eval swebench generate \
  --agent "${agent}" \
  --task-manifest "${manifest}" \
  --agent-name "${agent}_deepseek_v4_flash_smoke" \
  --num-workers 1 \
  --hours-per-task 5
