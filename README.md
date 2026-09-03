# External Agents Eval

This repository evaluates pinned external coding agents through benchmark-specific
backends. Pi Agent and Hermes Agent are supported on SWE-bench and DeepSWE v1.1;
Pier-native Codex is additionally supported on DeepSWE v1.1.

| Agent | Benchmark scope | Pinned version | Commit | Runtime |
|---|---|---|---|---|
| [Pi Agent](https://github.com/earendil-works/pi) | SWE-bench, DeepSWE | `v0.84.2` | `914cf1472e715297caa30db4b9535d534a9eb718` | Node 22.19 |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | SWE-bench, DeepSWE | `v2026.8.18` | `e624e9fde561e1add9388384012b295fde669ade` | Python 3.11 |
| [Codex CLI](https://github.com/openai/codex) | DeepSWE | `0.152.0` | — | Pier-installed Node 22 |

All agents use the direct DeepSeek provider and default to `deepseek-v4-flash`. The API key is
passed only as an environment variable; it is never written into an image, manifest, or
generated command record.

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
