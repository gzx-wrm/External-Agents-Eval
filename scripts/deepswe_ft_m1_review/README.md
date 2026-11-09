# Frozen ft_m1 + Mini review

`ft_m1_review` adds the existing paper pipeline's I/R → S → Repair/accept/hold
stages to a **fixed ft_m1 P0**. It does not resample A or Solver, modify the
frozen FT ledger, or supply the FT evidence packet. A later fresh-P0 experiment
can use a newly completed ft_m1 source through the same entry point.

## Adaptation

- Reuse `MiniV63RoleRunner` and the unchanged V6 controller, schemas, prompts,
  few-shot examples, source binding, durable SQLite state and stage-reuse protocol.
- I checks implementation obligations. R checks preservation obligations and
  diff impact, and may register additional regression obligations. R additions
  must be registered and assessed before scope completion, as in V6.
- I/R use native Bash to write Markdown records with structured assessment
  headers. Existing V6.3 file validation gives corrective feedback in the same
  conversation and submits a valid closed batch. File existence alone does not
  imply that a requirement is satisfied. S retains accept/hold/repair decisions.
- Each lane retains its own Mini conversation across rounds. S and Repair use
  the existing Mini backend and output contracts; no new finish tool is added.
- The `/testbed` baseline reference in inherited Mini guidance becomes `/app`.
  The original M1 Solver prompt and Bash loop are not changed.
- Native P0 includes the image environment baseline. Temporary Git indexes
  convert it to V6's baseline-relative delta and prove equal Git trees. After
  review, export a complete native-format patch again. Accept/hold must retain
  the exact original native patch hash. Keep original and converted hashes.
- Model settings remain the existing Mini V6.3 DeepSeek V4.1 Flash/high,
  thinking enabled, max output 65536. Review limits remain V6 defaults: batches
  of 3, up to 64 rounds, 45-minute lane deadline and 30-minute role timeout.
  Mini review invocations retain the existing 250-step limit. These are added
  review budgets; this experiment does not claim equal total cost to M1.
- Task containers keep the original image digest, M1 resource settings and
  native history/network isolation. Model credentials remain on the host.
  Official tests are injected only into a separate scoring container after
  model stages finish. Official scores never enter the review input.
  Images newly pulled by this run are removed after their worker ends;
  pre-existing images and images retained by preflight are preserved.
- Workflow errors remain explicit stage failures. They are not silently scored
  as an accepted or held P0. An incomplete paid role invocation is not resampled
  automatically; existing Mini invocation recovery rules remain in force.

## Usage

Run from an installed project checkout with mini-SWE-agent 2.4.6, Pier, Docker
and the existing model dependencies. Source directories must be the complete
server-side M1 bundles, including their frozen M0 runtime and grader references.
Use a new output directory. Repeat `--source-run` for the eight- and 22-task batches.

```bash
python scripts/deepswe_ft_m1_review/run.py prepare \
  --root /path/to/new-ft-m1-review \
  --source-run /path/to/ft_m1-failed8 \
  --source-run /path/to/ft_m1-passed22 \
  --task-id boa-hierarchical-evaluation-cancellation

python scripts/deepswe_ft_m1_review/run.py preflight --root /path/to/new-ft-m1-review

# Start only when explicitly requested. Credentials come from the host environment.
python scripts/deepswe_ft_m1_review/run.py run --root /path/to/new-ft-m1-review --workers 3
```

Omit `--task-id` during prepare to select all completed cases in the supplied
sources. Preparation freezes task/ledger/P0/image identities and source hashes.
Preflight checks every selected P0 conversion and actual container isolation
with **zero model calls**; it references, rather than reruns, M0's official
gold/control checks. It does not claim an online review smoke test.

Every worker reviews and immediately scores its final patch before taking
another task. Outputs include the frozen fork, P0 conversion proof, original
controller prompts, adapted Mini prompts, raw trajectories, per-obligation
Markdown notes, scope records, S decisions, route, P0/P1 hashes, final patch and
unchanged official verifier reports. Historical M0/M1 artifacts stay unchanged.

The batch runner preserves existing terminal results. Retrying an infrastructure
or stage failure requires a separately named run; it never automatically
resamples a normal reward=0 result.
