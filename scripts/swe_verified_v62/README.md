# Verified random100 / V6.2 / mini-SWE-agent

Remote root: `/root/v62-swe-verified100-20261026`.

This is a preparation bundle, not a running evaluation. The installed interpreter is
`venv312/bin/python` (Python 3.12), mini-SWE-agent is pinned to 2.4.6, and the classic
Verified-compatible official scorer is pinned to SWE-bench 4.1.0. SWE-bench 5.x uses
a different dataset format and is not used here.

Selection is a uniform random100 without replacement from all500 Verified test
instances, seed 20261026, dataset revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Keep this manifest for all baselines.
No task was selected, excluded or replaced based on a result.

All tool-enabled A/Solver/I/R/S/Repair loops run `minisweagent.agents.default.DefaultAgent`.
The existing V6.2 controller, contracts, FT compiler and review-note archival are reused.
The Pi helper class supplies file/worktree transport only; no Pi CLI or Pi agent is launched.
The FT compiler remains an independent, tool-free request with the original prompt and protocol.

The Mini Bash helper `ft_read` produces exact file excerpts and records the same output
that the model sees. The unchanged FT functions then verify and merge only those returned
baseline lines. Search reasoning, other shell outputs and the initial A draft are not
passed to the compiler. `mini.json` preserves native messages; `trajectory.json` records
the explicit Bash-to-read projection used by the FT collector.

Reviewer history is retained between batches; completed submission tool calls are closed
before reusing the conversation. Format-correction stages have only a literal
`submit_json` helper, without filesystem or shell execution.

## Isolation

- Docker containers use `--network none`, no privileged mode, all capabilities dropped,
  no-new-privileges, no host mounts and no Docker socket. Each task has 2 CPUs/6 GiB RAM.
- Model calls run on the host. API keys are not forwarded into Docker exec or task containers.
- Before any agent action, the original repository is replaced by a single deterministic
  synthetic commit. Only the selected baseline's trees/blobs are retained; object IDs are
  checked against that exact allowlist. Historical commits, unreachable solution blobs,
  branches, remotes, reflogs and alternates do not survive.
- Other `.git` directories found under `/testbed`, `/root` and `/opt` are removed from
  the disposable task container before agent access.
- The baseline tree is unchanged. Original benchmark commit IDs remain in host metadata
  for the independent scorer; generated patches are ordinary diffs against the same files.
- Agent input is `manifests/public_tasks.jsonl`, containing only five public fields.
  Gold patches, test patches, hints and test outcome lists stay in the host's protected
  `grader/reference_tasks.json` and are never mounted or uploaded to agent containers.
- Generation and official evaluation use separate containers. Hidden tests are injected
  only by the scorer after the agent has finished.

## Checks and launch commands

```bash
cd /root/v62-swe-verified100-20261026/project
../venv312/bin/python scripts/swe_verified_v62/run.py plan \
  --root /root/v62-swe-verified100-20261026
```

`preflight/readiness.json` records the actual checks; the first sampled official image
was pulled and tested. The `preflight/offline-pipeline/` trace uses stub responses and
does not count as a model smoke run or solved benchmark task.

For the subsequent real one-task smoke run, supply `DEEPSEEK_API_KEY` in the host
environment (no credential is stored in this bundle):

```bash
../venv312/bin/python scripts/swe_verified_v62/run.py generate \
  --root /root/v62-swe-verified100-20261026 \
  --task-id astropy__astropy-13453 --workers 1
```

Then generate the same fixed100 by removing `--task-id` and setting `--workers 2`.
Do not start this during preparation. Every task preserves per-stage prompts,
native Mini traces, the evidence packet, FT request/response, ledger, patches and failures.

Newly pulled generation images are removed after their task; pre-existing images are
left untouched. This bounds disk growth on the 140-GiB-free server. Pulls stop if free
space falls below 15 GiB. No global Docker prune is used.

Scoring is a separate command with a fresh run ID:

```bash
bash scripts/swe_verified_v62/evaluate.sh \
  /root/v62-swe-verified100-20261026 verified100-v62-smoke-r1
```

The official scorer consumes only submitted predictions and host-side reference data.
Keep empty patches, generation failures, infrastructure errors, unresolved tasks and
resolved tasks separate when reporting results. A preparation check has no reward.
