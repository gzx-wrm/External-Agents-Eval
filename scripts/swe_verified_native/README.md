# Native mini-SWE-agent / Verified random100

Prepared host: `root@223.109.239.36:12748`.
Experiment root: `/root/mini-swe-verified100-20261027`.

Use the same frozen Verified random100, seed 20261026, dataset revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. This baseline includes both smoke
instances; it does not resample or filter tasks based on previous results.

## Approved configuration

- mini-SWE-agent 2.4.6, LiteLLM 1.106.4, Python 3.12.15; official SWE-bench scorer 4.1.0.
- Unmodified `DefaultAgent`, `DockerEnvironment`, native Bash schema and action parser.
- Official `config/benchmarks/swebench.yaml` system/instance prompts, observation
  template and format-error template, copied without changing their string values.
- DeepSeek V4.1 Flash (`deepseek/deepseek-flash`), high reasoning, thinking enabled,
  max output 65536 tokens; 250 steps and 10800 seconds per agent, no monetary stop threshold.
- Preserve the official 60-second command timeout, Bash interpreter, BASH_ENV
  activation, and 10000-character observation clipping. Do not substitute V6.2's
  solver prompts or tool-output formatting.
- API request timeouts and retries retain native defaults; no additional V6.2
  request-level timeout or retry overrides are added.
- The external model subclass only normalizes an explicitly empty assistant
  `reasoning_content` to one space in outgoing requests, as in V6.2. Missing/None
  values are not synthesized; original trajectory messages retain their values.
- No A, FT, ledger, reviewer, summarizer or repair stages; no Pi tools or agent loop.

The runner adds experiment bookkeeping and the shared isolation protocol before
calling the native agent. Each task gets 2 CPUs/6 GiB, network=none, no mounts or
forwarded host credentials, and a single synthetic baseline commit. Other Git
histories in the image are removed. Model calls happen on the host.

Images are pulled before the native Docker environment starts so cold downloads
do not consume its upstream startup timeout. Container lifetime is four hours to
cover preparation and the approved three-hour agent window. This does not extend
the agent's 10800-second budget. In the combined pipeline, newly pulled images are
retained through that task's official evaluation and then removed. Pre-existing
images and preflight images are retained. No global prune.

The final prediction is the native agent's submitted text, as in the official
runner. There is no fallback that repairs the patch or exports additional files
after a timeout. Preserve exit status, empty patches and infrastructure failures
separately; a preparation check is not a solved benchmark instance.

## Preparation checks (no model API calls)

```bash
ROOT=/root/mini-swe-verified100-20261027
"$ROOT/venv312/bin/python" "$ROOT/scripts/run.py" prepare --root "$ROOT"
"$ROOT/venv312/bin/python" "$ROOT/scripts/run.py" offline-check --root "$ROOT" \
  --task-id astropy__astropy-13453
```

The offline check uses simulated API responses with the real native model adapter,
agent loop and isolated Docker environment. It checks submission, the transport
compatibility fix and preservation of the original reasoning field. It does not
create a benchmark prediction or run the official grader.

Formal concurrency and launch require separate user confirmation. Generation
requires an explicit output directory, worker count and a host-only
`DEEPSEEK_API_KEY`. No API credential is written to this preparation bundle.
Host-side gold data stays under `grader/` and is never provided to generation or
uploaded into task containers.

## Per-slot solving and official evaluation

Use `solve-evaluate` for the formal baseline, with explicit `--output`, `--workers`
and `--run-id`. Each worker completes one whole sequence before taking another task:

1. Run the unchanged native agent and save its P0 submission and raw trajectory.
2. Remove the generation container.
3. Call the installed official SWE-bench `run_instance` evaluator in a fresh
   container, reusing the unmodified task image. The test timeout remains the
   official default of 1800 seconds. Test results stay on the host.
4. Preserve the official report, clean up the task's newly pulled image, and take
   the next task. The maximum simultaneous official evaluations equals the number
   of pipeline workers; there is no extra independent scoring pool.

Set the host working directory to the run's `verifier/` once before starting the
workers. The wrapper changes neither the installed official grader nor its tests.
Generation retains its approved 2-CPU/6-GiB limits; official scoring containers use
the installed official evaluator's defaults.

The `--output` directory contains per-task `result.json`, `prediction.json`,
`evaluation.json` and `progress.json`. Its parent contains `monitor-status.json`,
`evaluation-config.json` and `verifier/`. Keep one output parent and official run
ID per experiment. Completed evaluation receipts are reused only for the same P0
and run ID; empty submissions, generation errors, grading errors and unresolved
official reports remain distinct. The final aggregate uses the official
`make_run_report` function.
