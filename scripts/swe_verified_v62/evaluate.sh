#!/usr/bin/env bash
set -euo pipefail
experiment_root="${1:-/root/v62-swe-verified100-20261026}"
evaluation_id="${2:?Provide a fresh evaluation run ID}"
cd "$experiment_root"
exec "$experiment_root/venv312/bin/python" -m swebench.harness.run_evaluation \
  --dataset_name "$experiment_root/grader/reference_tasks.json" \
  --predictions_path "$experiment_root/outputs/predictions.jsonl" \
  --run_id "$evaluation_id" \
  --max_workers 2 \
  --namespace swebench \
  --cache_level env \
  --clean False
