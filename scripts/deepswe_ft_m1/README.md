# ft_m1: frozen FT ledger + native Mini

The initial pilot runs only the eight valid reward=0 tasks from native Mini M0.
The follow-up batch runs the complementary 22 M0 reward=1 tasks, with the same
frozen ledger-input protocol, model, budgets and solve/score loop. Each batch has
its own directory and selection; the original eight outcomes stay unchanged.
It starts each case from the original repository image, without reusing the M0 patch
or conversation. The sole model-visible change is the original FT pre-Solver
ledger and the approved brief guidance appended to the initial task text.

The complete `ledger` object is extracted from the historical
`v6_review_repair/v6/pre-solver-ledger.json`, including O/P/Q, summary and source
references. The evidence packet, source-binding payload, Reviewer outputs, failed
tests and manual corrections are excluded. Original degraded ledgers stay unchanged.
Source hashes, task checksums, original image digests and M0 scores are frozen in
`selection.json`; each model input is recorded under `inputs/<task>/task.json`.

The independent experiment bundle contains a byte-for-byte copy of the corrected
M0 runner as `scripts/m0_run.py`, plus its unchanged native model compatibility
helper and repository sanitizer. No M0 source or result is modified.

DeepSeek V4.1 Flash/high, thinking enabled, max_tokens=65536, 10800 seconds per
task, no step/cost limit, native Bash and all original Mini prompt templates are
unchanged. Three pipeline slots each solve and then immediately run the original
official DeepSWE verifier. Task containers remain offline with sanitized Git,
2 CPUs and 8192 MiB. Scoring is isolated from model generation.

Preflight checks all eight frozen ledgers and configuration equality, then runs
the actual native Mini protocol, task injection and patch export against a real
original task container with simulated API responses (zero model API calls).
The unchanged official scorer's passed M0 gold/control checks are referenced
explicitly, rather than reported as newly executed checks.

This failure-selected pilot measures recovery on eight M0 failures; it does not
measure regressions on the other 22 tasks or establish a new 30-task pass@1.
