# External Agents Eval

This repository evaluates pinned external coding agents through benchmark-specific
backends. Pi Agent and Hermes Agent are supported on SWE-bench and DeepSWE v1.1;
Pier-native Codex is additionally supported on DeepSWE v1.1.

| Agent | Benchmark scope | Pinned version | Commit | Runtime |
|---|---|---|---|---|
| [Pi Agent](https://github.com/earendil-works/pi) | SWE-bench, DeepSWE | `v0.84.2` | `914cf1472e715297caa30db4b9535d534a9eb718` | Node 22.19 |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | SWE-bench, DeepSWE | `v2026.8.18` | `e624e9fde561e1add9388384012b295fde669ade` | Python 3.11 |
| [Codex CLI](https://github.com/openai/codex) | DeepSWE | `0.152.0` | — | Pier-installed Node 22 |

All agents default to the direct DeepSeek provider and `deepseek-v4-flash`. API keys are passed
only as environment variables; they are never written into an image, manifest, or generated
command record.

V5/V6 Pi workflows can assign a provider, model and optional endpoint to each internal stage
with `deepswe run --stage-model-config path/to/stage-models.json`. The file is secret-free; API
keys are still supplied through environment variables. For example:

```json
{
  "solver": {"model": "deepseek-v4-flash"},
  "A": {"model": "openai/gpt-5.6-sol", "base_url": "https://api.openai.com/v1"},
  "I": {"model": "openai/gpt-5.6-sol", "base_url": "https://api.openai.com/v1"},
  "R": {"model": "openai/gpt-5.6-sol", "base_url": "https://api.openai.com/v1"},
  "S": {"model": "openai/gpt-5.6-sol", "base_url": "https://api.openai.com/v1"},
  "repair": {"model": "deepseek-v4-flash"},
  "json_corrector": {"model": "openai/gpt-5.6-sol", "base_url": "https://api.openai.com/v1"}
}
```

Supported built-in providers are `deepseek`, `openai`, and `zai`. Set `DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`, and/or `ZAI_API_KEY` according to the stages used. When omitted, every stage inherits the command's
`--model` value and the existing DeepSeek behavior is unchanged. The checked-in V6 experiment
configuration at `configs/v6-deepseek-solver-openai-review.json` uses the OpenAI-compatible
review endpoint for A/I/R/S and JSON correction while keeping Solver and Repair on DeepSeek.
For a run that uses both providers, `scripts/run_with_provider_keys.sh` injects both keys only
for the child process.

The V6.1 all-GLM configuration at `configs/v6.1-glm-5.3.json` routes every internal stage
through Pi's native `zai/glm-5.3` provider and the standard BigModel API endpoint. Use
`scripts/run_with_zai_key.sh` to pass the Z.AI credential to the child process without storing
it in the configuration or job metadata.

The V6.1 DeepSeek V4.1 Flash configuration at
`configs/v6.1-deepseek-v4.1-flash.json` registers the newer model under Pi's native
`deepseek` provider using the OpenAI-compatible Chat Completions wire API. The provider's wire
model ID is `deepseek-flash`; its display metadata identifies the V4.1 Flash model. Its model metadata
is emitted into the per-role `models.json` because Pi 0.84.2 does not include this model in
its built-in catalog. Verify the provider's published context and output limits before a
large run; the checked-in values are the initial V4.1 Flash integration defaults.

```bash
external-agents-eval deepswe run --agent pi-v5-review-repair \
  --stage-model-config configs/v5-openai-review.json
```

## Repository structure

```text
external_agents_eval/
  agents/                    # Agent identity and reusable Docker adapters
    pier/                    # Pi/Hermes plugins for the official Pier runner
  benchmarks/
    swebench/                # Fixed manifest, image audit, generation and judge
    deepswe/                 # v1.1 source pin, sampling, Pier runner and summary
  cli.py                     # Grouped CLI plus old SWE-bench aliases
docker/                      # Reusable SWE-bench runtime images
scripts/                     # Secret-safe launch helpers
tests/
```

Benchmark code owns task discovery, isolation and scoring. Agent code owns the upstream
version, invocation, trajectory conversion and provider access. Adding a benchmark no
longer requires changing the SWE-bench runner, and adding an agent does not require
copying benchmark logic.

## Installation

The SWE-bench backend retains the existing Python 3.10-compatible environment:

```bash
conda activate hgm
cd /Users/gzx/code/python/external_agents_eval
python -m pip install -e '.[dev]'
```

DeepSWE uses the pinned `datacurve-pier==0.3.1`, which requires Python 3.12 or newer. Keep it in a
separate environment:

```bash
python3.12 -m venv .venv-deepswe
source .venv-deepswe/bin/activate
python -m pip install -e '.[deepswe,dev]'
```

## DeepSWE v1.1: fixed random 60

The supported v1.1 source is the public DeepSWE `main` commit
`0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea` (113 tasks). `prepare` clones that exact
commit, validates that all tasks use the separate verifier and v1.1 images, then samples
from sorted task IDs with Python's seeded shuffle. The resulting manifest records the
source commit, population fingerprint, seed and all 60 selected task IDs.

```bash
external-agents-eval deepswe prepare \
  --checkout-dir .benchmarks/deep-swe \
  --sample-size 60 \
  --sample-seed 20260901
```

Run the agents against that same immutable manifest. Start locally with one worker:

```bash
scripts/run_with_deepseek_key.sh \
  external-agents-eval deepswe run \
  --agent pi \
  --num-workers 1

scripts/run_with_deepseek_key.sh \
  external-agents-eval deepswe run \
  --agent hermes \
  --num-workers 1

scripts/run_with_deepseek_key.sh \
  external-agents-eval deepswe run \
  --agent codex \
  --num-workers 1
```

DeepSWE does not impose an evaluation-side Hermes turn limit unless `--max-turns` is
provided explicitly. When it is omitted, the adapter supplies Hermes with a practically
unreachable integer iteration limit and Pier's per-task timeout remains the effective
stop condition. The adapter also leaves the model's maximum output tokens unspecified,
so the native Hermes/DeepSeek default applies.

The Codex path uses Pier 0.3.1's built-in `codex` agent and pins
`@openai/codex@0.152.0`. Its checked-in TOML selects DeepSeek's native Responses API
and reads `DEEPSEEK_API_KEY` from the environment; no Codex-specific adapter or API
proxy is used.

Use `--dry-run` to inspect the exact Pier command and selected IDs without starting a
job. For an AMD64 Linux or configured Modal environment, increase concurrency as
resources permit; Modal can be selected with `--environment modal`.

DeepSWE v1.1 runs the agent in `/app` without task-test access. Runtime egress is filtered
to `api.deepseek.com` only. The Pi/Hermes plugins stage and commit all repository changes;
all 60 selected task instructions also explicitly require Codex to commit. This matters
because the separate verifier grades only `base_commit..HEAD`. Confirm that `HEAD`
advances in a one-task Codex smoke run before starting all 60 tasks. The full binary patch
is preserved; SWE-bench's test-diff filter is not applied.

Pier writes trials under `results/deepswe/<job-name>/`. Aggregate the official reward
records with:

```bash
external-agents-eval deepswe summary \
  --job-dir results/deepswe/deepswe-v1.1-pi-random60-seed20260901
```

### Specification review and one-pass repair experiment

The `pi-review-repair` composite Agent provides versioned, reversible review contracts. It
replays or solves P0, builds an obligation ledger in a clean worktree, assesses P0 in a
separate Pi session, and either preserves P0 or runs one repair session. Pier receives and
verifies only the committed P1.

`--review-arm` selects the control behavior. Its historical default contract is preserved;
`--review-contract-version 3.0` explicitly selects the V3 obligation-level bundle for E1/E2:

| Arm | Contract | Behavior |
|---|---|---|
| `E0` | v1.0 | Frozen baseline from `docs/第一版实验方案.md` |
| `E1` | v2.0 | Criterion-level sources, Stage B challenges, effective review set, and `abstain` |
| `E2` | v2.0 | E1 plus mandatory Repair audit and audit-failure fallback to P0 |

The v2 implementation follows `docs/第二版实验方案.md`, and the opt-in V3 implementation
follows `docs/第三版实验方案.md`. E0 keeps the historical v1 Schema, prompts, few-shots,
routing, and result readability intact. E1/E2 continue to default to V2 so old commands are
reproducible; either may use V3 explicitly, while E2 still means mandatory Repair audit.

Freeze the exact source job, source manifest, base commits, trial IDs, and P0 digests before
running. This command reads only source configuration and `artifacts/model.patch`; it does
not open verifier, hidden-test, or solution artifacts:

```bash
external-agents-eval deepswe prepare-review-manifest \
  --source-job-dir results/deepswe/<exact-source-job> \
  --checkout-dir .benchmarks/deep-swe \
  --scope random60 \
  --output manifests/deepswe_v1.1_pi_random60_p0_20260901.json
```

Use `--scope near-miss16` for the original exploratory 16-task recovery set. The
`random60` scope freezes all 60 source tasks. Because the two reference-assisted
few-shot development tasks are also members of that source sample, they are still run for
pipeline completeness but are marked as excluded from the primary metrics. The resulting
main analysis therefore has 58 tasks; `abs-stepped-slices` and
`effect-sse-httpapi-streaming` are reported separately as development-contaminated cases.

Formal existing-P0 runs use the explicitly authorized reference-assisted development set at
`few_shots/reference_assisted_v1`. It contains four Reviewer JSON artifacts under each of
`abs-stepped-slices/` and `effect-sse-httpapi-streaming/`, plus a root `provenance.json` that
records the mode and source digests. Reference materials were used offline to audit labels but
are not embedded in Reviewer-visible examples. At launch the loader validates all nine files,
copies them to a content-addressed read-only snapshot, records every digest in the batch run
record, and makes each trial revalidate the aggregate digest. The composite workflow also fixes
task-level attempts at one; use concurrency, not automatic retries, for either experiment scope.

Regenerate the exact set only while the pinned development sources retain their recorded hashes:

```bash
.venv-deepswe/bin/python scripts/build_reference_few_shots.py \
  --output few_shots/reference_assisted_v1
```

```bash
scripts/run_with_deepseek_key.sh \
  external-agents-eval deepswe run \
  --agent pi-review-repair \
  --review-arm E0 \
  --p0-manifest manifests/deepswe_v1.1_pi_random60_p0_20260901.json \
  --few-shot-dir few_shots/reference_assisted_v1 \
  --job-name deepswe-v1.1-pi-review-repair-random60 \
  --num-workers 5
```

For v2 development or a dry-run preflight, use the unrelated synthetic examples in
`few_shots/source_grounded_v2`:

```bash
external-agents-eval deepswe run \
  --agent pi-review-repair \
  --review-arm E2 \
  --p0-manifest manifests/deepswe_v1.1_pi_random60_p0_20260901.json \
  --few-shot-dir few_shots/source_grounded_v2 \
  --job-name deepswe-v1.1-pi-review-repair-v2-smoke \
  --task-name <task-id> \
  --dry-run
```

E1 uses the same v2 synthetic bundle but does not make Repair audit validity a publication
gate. E0 rejects v2 few-shots, and E1/E2 reject v1 few-shots. A dry run validates and prints
the configuration without starting Pier, an experiment, a container, or an image build.

For V3, use the independent obligation-level synthetic bundle and select the contract
explicitly:

```bash
external-agents-eval deepswe run \
  --agent pi-review-repair \
  --review-arm E2 \
  --review-contract-version 3.0 \
  --p0-manifest manifests/deepswe_v1.1_pi_random60_p0_20260901.json \
  --few-shot-dir few_shots/source_grounded_v3 \
  --job-name deepswe-v1.1-pi-review-repair-v3-smoke \
  --task-name <task-id> \
  --dry-run
```

V3 replaces criterion-level review with atomic Oxx/Lxx obligations, Controller-bound task
and base-repository sources, unified source/implementation assessments, automatically
expanded repair-request contexts, and obligation-level Repair audits. The V3 few-shots are
synthetic and contain no benchmark task or evaluation answer.

Before the full batch, the same frozen configuration can be exercised on one task by adding
`--task-name <task-id>` and using a distinct smoke-test job name.

Each trial records the append-only state log, rendered prompts, Schema/few-shot/prompt
digests, raw and validated Reviewer outputs, P0/P1, resource usage, and the final workflow
manifest under `review_repair/`. If a Reviewer role or JSON contract fails, P1 falls back to
P0; a Reviewer worktree mutation is recorded as a protocol violation and cannot propagate
to the repair worktree. `COMPLETED` in that manifest denotes completion of the composite Agent;
the subsequent verifier lifecycle and reward remain Pier-owned in the trial `result.json`.
Reviewer prompts require one fenced `json` code block. The controller normalizes responses in
three conservative stages: exact JSON, one valid fenced object, then one unambiguous embedded
top-level object decoded with Python's JSON parser. It never uses greedy brace matching, and
ambiguous responses continue through the bounded correction path.

V2 adds `inputs/source_policy.json`,
`reviewer/source_anchor_validation.json`, `reviewer/effective_review_set.json`, and
`reviewer/routing_validation.json`. E2 also records raw, validated, or incomplete Repair
audit artifacts below `repair/`, plus `repair/repair_sequence_observation.json`. That
observation records whether the audit Schema and frozen inputs were read before the first
explicit `edit`/`write` tool call; it deliberately marks shell activity as ambiguous and is
not a tool-level edit gate or proof of semantic reasoning. The writable audit target is
materialized outside the Git
worktree, so it cannot enter `artifacts/model.patch`. After a normally completed Repair,
missing or invalid audit falls back to P0 while preserving the candidate patch only as a
diagnostic artifact. A Repair timeout remains an agent failure and may publish a safely
captured partial patch for verifier scoring under the existing three-hour policy.

V3 replaces the source-policy artifact with `reviewer/source_bindings.json` and, after
Stage B, `reviewer/additional_source_bindings.json`. Repository anchors are enriched by the
Controller with an exact base quote, resolved revision, and file SHA-256; models provide only
the base-relative path and line range. The manifest records the selected contract, prompt
bundle, few-shot summary, Schema summary, and all corresponding content hashes.

V2/V3 manifests hash the workflow package and the Pi/DeepSWE runtime adapters that select,
execute, aggregate, and report the experiment. These are content hashes of the actual files,
so an uncommitted working tree is not represented merely by `git HEAD`. E0 retains the
historical implementation-hash shape and all v1 contract paths.

V5 is a separate agent entry and does not accept the V1--V4 arm, contract, prompt, or
few-shot options. It runs A, independent I/R sessions, S, and at most one Repair against the
same frozen P0 manifest. Start with a one-task preflight, then remove `--dry-run` and wrap the
command with `scripts/run_with_deepseek_key.sh` for the real smoke run:

```bash
external-agents-eval deepswe run \
  --agent pi-v5-review-repair \
  --p0-manifest manifests/deepswe_v1.1_pi_random60_p0_20260901.json \
  --job-name deepswe-v1.1-pi-v5-review-repair-smoke \
  --task-name <task-id> \
  --num-workers 1 \
  --dry-run
```

V5 stores its SQLite state, content-bound role exports, immutable review-note bundles,
P0/P1 patches, and decision under the trial's `v5_review_repair/` artifact directory. Keep
task-level attempts at one; the workflow owns its bounded role retries and resume identities.
After the task image is built, V5 records any environment-authored Git diff as the baseline,
materializes that baseline into every isolated role worktree, and stores P0/P1 relative to
the baseline. `manifest.json` binds the original base commit, baseline digest, and patch
coordinate system. Historical base-relative P0 manifests are normalized before review.

V6.1 runs A before Solver, appends the frozen O/P/Q ledger to the original coding
instruction, and uses that ledger again in I/R → S → Repair review. A structurally invalid
ledger ends the run before Solver; source-anchor problems are recorded as degraded quality.
The normal run generates P0 and does not accept `--p0-manifest`.

```bash
external-agents-eval deepswe run \
  --agent pi-v6-review-repair \
  --job-name deepswe-v1.1-pi-v6-review-repair-smoke \
  --task-name <task-id> \
  --stage-model-config configs/v6-deepseek-solver-openai-review.json \
  --num-workers 1 \
  --dry-run
```

V6 stores its run artifacts under `v6_review_repair/`, including the pre-Solver ledger,
Solver prompt/trajectory, generated P0, reused V5 review artifacts, and final delivery.

V6.2 is an additive entrypoint, `--agent pi-v62-review-repair`. It runs the original A
exploration, freezes the exact baseline lines returned by its `read` calls, and uses the
unchanged FT prompt and single-call inference protocol to generate the ledger for Solver.
The original A draft is retained only as a diagnostic; Solver/I/R/S/Repair use the FT ledger.
Artifacts remain under `v6_review_repair/`, with `v6.2/evidence.json`,
`v6.2/evidence-search/`, and `v6.2/ledger/` added. Both A steps use the existing `A` model
route. FT retains thinking/high, 65,536 output tokens including reasoning, a 1,800-second
request timeout, and no automatic inference retries. See
[the V6.2 workflow notes](docs/V6.2流水线说明.md) for the preserved protocol and validation.

To retry a V6.1 stage in a new one-task job, create a JSON fork plan. `source_run` is the
old trial's `v6_review_repair/` directory; `reuse` can override an individual predecessor
with another trial directory. For a manually edited A ledger, set `reuse.A` to the edited
`ledger.json` file and start at `Solver`. The supported `start_at` values are `A`, `Solver`,
`I/R` (both reviewers), `I`, `R`, `S`, `Repair`, and `Verifier`. All stages from the selected
point run through the normal Pier verifier. The plan checks task/base/baseline identity and
the selected ledger, P0, review and P1 hashes before reuse.

```json
{
  "task_id": "<task-id>",
  "start_at": "I",
  "source_run": "results/deepswe/<old-job>/<trial>/v6_review_repair",
  "reuse": {"R": "results/deepswe/<other-job>/<trial>/v6_review_repair"}
}
```

```bash
external-agents-eval deepswe run \
  --agent pi-v6-review-repair \
  --manifest <same-manifest-as-source> \
  --task-name <task-id> \
  --job-name <new-job-name> \
  --fork-plan /absolute/path/to/fork.json
```

Paths in the plan are absolute or relative to the plan file. Existing V6.1 results can be
used as sources; new runs also write `v6/stages/*.json` for intermediate stage reuse.

To retry several tasks in one concurrent Pier job, create a JSON mapping from each task ID
to its own fork plan. Paths in this mapping are relative to the mapping file (or absolute).
Each plan's `task_id` must match its mapping key. The mapping selects the tasks when
`--task-name` is omitted; if provided, the selected task IDs must match the mapping.

```json
{
  "<task-a>": "task-a/fork.json",
  "<task-b>": "task-b/fork.json"
}
```

```bash
external-agents-eval deepswe run \
  --agent pi-v6-review-repair \
  --manifest <same-manifest-as-sources> \
  --job-name <new-batch-job-name> \
  --fork-plans /absolute/path/to/plans.json \
  --num-workers 4
```

Pier runs the selected tasks concurrently in one job, so the Docker job uses one BuildKit
builder while each trial reads its own ledger and upstream stage outputs.

After the frozen batch has finished, compute the experiment-specific metrics separately
from the ordinary benchmark summary:

```bash
external-agents-eval deepswe review-summary \
  --job-dir results/deepswe/<review-job> \
  --p0-manifest manifests/deepswe_v1.1_pi_random60_p0_20260901.json \
  --details
```

For `random60`, this reports paired P0/P1 pass rates, net pass change, `fail → pass`,
`pass → fail`, retention, recovery, and the two excluded-task transitions. V2 summaries also
include approval coverage, abstain rate, source-anchor status, challenge dispositions,
effective/unresolved criteria, Repair audit status, source-check decisions, request closure,
Repair sequence observation status, and per-role timeout/cost data. For
V3, the same report adds obligation source and implementation statuses, effective/nonbinding/
unresolved obligation counts, late-obligation counts, and the corresponding Stage A omission
count. For
`near-miss16`, it retains the fixed-denominator recovery analysis and deliberately labels
that result as exploratory rather than an unbiased overall pass-rate uplift. Both modes also
report Schema validity, verdicts, obligation categories, F2P recovery, P2P regression, and
patch-size changes.

The DeepSWE task images are currently `linux/amd64`. On an ARM Mac, a one-task Docker
smoke run may use emulation, but the full 60-task evaluation should run on AMD64 Linux or
Modal. Local Docker runs use a job-scoped BuildKit builder. Pier removes each trial's
containers, networks, volumes and Compose-built images, then prunes unused cache only
from that dedicated builder; the runner removes the builder after the job exits. The
default Docker builder and unrelated project data are never pruned. On the first Ctrl-C,
the runner waits for Pier to finish this cleanup before exiting.

## SWE-bench

The SWE-bench backend still reuses the exact audited task-image IDs already present on
this machine. Generation never pulls, builds or mutates those task images. It copies an
architecture-matched pinned agent runtime into each disposable `/testbed` container and
collects `git diff --binary` in the standard prediction schema.

The grouped commands are:

```bash
external-agents-eval agents build-runtimes --agent all
external-agents-eval swebench audit-task-images

scripts/run_with_deepseek_key.sh \
  external-agents-eval swebench generate \
  --agent pi \
  --num-workers 3 \
  --hours-per-task 5 \
  --resume

external-agents-eval swebench status --agent-dir /absolute/path/to/agent-result
external-agents-eval swebench report \
  --agent-dir /absolute/path/to/agent-result \
  --num-workers 3 \
  --report-retries 2
```

The former flat commands (`build-runtimes`, `audit-task-images`, `generate`, `report`,
and `status`) remain as compatibility aliases.

## Result contracts

SWE-bench keeps its existing result layout and standard prediction JSON. DeepSWE keeps
Pier's official job/trial layout, including:

```text
<trial>/
  agent/
    trajectory.json          # ATIF-v1.7
    pi.jsonl | hermes-output.json | codex.txt
  verifier/
    reward.json
    ctrf.json
    test-stdout.txt
    run.log
    reports/
  artifacts/model.patch
  results.json
```

Do not commit `.runtime_cache`, `.benchmarks`, result directories, credentials, or copied
agent runtimes.
