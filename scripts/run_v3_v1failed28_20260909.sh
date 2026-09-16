#!/usr/bin/env bash
# Independent V3 E2 rerun of all 28 unsolved tasks in the final V1 58-task aggregate.
# Selection: primary58 overridden by canary3/recovery11, verifier reward == 0.
# Only task IDs are passed to agents; prior verdicts and verifier evidence are not.
# Usage: bash scripts/run_v3_v1failed28_20260909.sh --dry-run
# Actual run: supply DEEPSEEK_API_KEY via scripts/run_with_deepseek_key.sh.
set -euo pipefail
umask 077

task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_project_root"
export PYTHONUNBUFFERED=1

exec .venv-deepswe/bin/python -m external_agents_eval.cli deepswe run \
  --agent pi-review-repair \
  --checkout-dir .benchmarks/deep-swe \
  --manifest external_agents_eval/benchmarks/deepswe/data/deepswe_v1.1_random60_seed20260901.json \
  --jobs-dir results/deepswe \
  --job-name deepswe-v1.1-pi-review-repair-v3-v1failed28-w4-20260909 \
  --model deepseek-v4-flash \
  --p0-manifest manifests/deepswe_v1.1_pi_random60_p0_20260901.json \
  --few-shot-dir few_shots/source_grounded_v3 \
  --review-arm E2 \
  --review-contract-version 3.0 \
  --workflow-role-timeout 10800 \
  --num-workers 4 \
  --attempts 1 \
  --environment docker \
  --task-name bandit-interprocedural-taint-checks \
  --task-name clack-async-autocomplete-options \
  --task-name claude-code-by-agents-recursive-delegation \
  --task-name expr-try-catch-errors \
  --task-name fastapi-deprecation-response-headers \
  --task-name geo-shapeindex-serialization \
  --task-name gql-incremental-graphql-delivery \
  --task-name happy-dom-deterministic-intersectionobserver \
  --task-name httpx-deterministic-cookie-store \
  --task-name kombu-virtual-queue-dead-lettering \
  --task-name koota-pair-relation-tracking \
  --task-name langchain-request-coalescing \
  --task-name meriyah-explicit-resource-declarations \
  --task-name obsidian-linter-auto-table-of-contents \
  --task-name obsidian-linter-link-format-conversion \
  --task-name opa-template-string-reconstruction \
  --task-name returns-validated-error-accumulation \
  --task-name sqlfmt-create-table-ddl-formatting \
  --task-name sqlite-utils-safe-import-checkpoints \
  --task-name tengo-callable-instance-isolation \
  --task-name termenv-preserve-ansi-resets \
  --task-name testem-bail-on-test-failure \
  --task-name textual-kitty-key-phases \
  --task-name updo-policy-alerting \
  --task-name vulture-persistent-analysis-cache \
  --task-name wazero-multi-module-snapshots \
  --task-name yaegi-go-embed-directives \
  --task-name ytt-jsonpath-query-api \
  "$@"
