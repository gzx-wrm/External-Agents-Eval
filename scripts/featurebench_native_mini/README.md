# Native Mini × FeatureBench Lite, CPU-validated mode

Official data: `LiberCoders/FeatureBench`, v1.1 Lite, revision
`76b4a4566e04f4bcc13c35125d4f301791efa736`; 30 instances, 26 Lv1 and 4 Lv2.
Official code: `8d4e347ec57546685c5a87e8676bf575db022ea6`.

The user approved testing CPU eligibility first and deferring actual GPU
dependencies. Nineteen task runtime configurations request GPU; this flag alone
does not establish whether the original tests can run on CPU.

The stock mini-SWE-agent 2.4.6 `DefaultAgent`, `DockerEnvironment`, native Bash,
generic `mini.yaml` prompts and observation/error templates are used. Model:
DeepSeek V4.1 Flash/high, thinking enabled, max_tokens=65536. Each task gets
10800 seconds of agent time, no step/cost stop, 2 CPUs and 6 GiB. Each Bash call
retains Mini's 30-second default. The only model transport compatibility helper
is the previously approved outbound empty-reasoning fix. No ledger/reviewer/repair.

Before the model starts, official `_initialize_level1` or `_initialize_level2`
and package-cache cleanup run, without terminal-recorder installation. All Git
history is sanitized after this initialization, preserving the actual task tree.
The model sees only the original public problem statement. No hidden tests,
reference patch, CPU-validation results, API credentials or host mounts enter
its container. Conda activation is an environment-level Bash interpreter prefix,
not a new model tool or an edited prompt. Raw native trajectories are preserved.

Each slot performs CPU reference/control validation, one model solve, and
immediate official scoring. For Lv1 the CPU reference is the official gold fix.
Official gold mode excludes Lv2; therefore Lv2 CPU validation runs the same
tests against the pristine original implementation through a separate diagnostic
pytest invocation, with no masking or F2P-test deletion. Its JUnit report must
contain actual executed tests and no failures; empty/all-skipped/timed-out runs
are not readiness. This probe is never reported as a Lv2 benchmark score.
Actual model generation and scoring retain the real Lv2 empty-repository protocol.
Only a positive reference with actual test results and a failing negative control
qualify a task for paid generation. GPU dependency, other environment failures,
and scored model failures are recorded separately.

All scoring uses the unedited official `run_instance` and test/report logic.
Its container factory is wrapped only for the approved CPU hardware mode and
resource isolation; this applies equally to reference, control and model.
Tests requiring unavailable network/build dependencies remain infrastructure
failures instead of scored model failures.

The user authorized formatting the blank 50 GiB `/dev/vdb` as XFS and mounting
it at `/mnt/featurebench`. It is mounted for this session; `/etc/fstab` is not
modified. A separate Docker engine uses `/mnt/featurebench/docker`, its own
socket/namespace, and no bridges, NAT or host firewall changes. The old engine,
historical images and experiment artifacts stay intact. Images run in groups;
after all tasks and scores in a group terminate, only that new engine's group
image is removed to keep the 50 GiB filesystem usable. Up to three slots run.

Server root: `/root/native-mini-featurebench-lite-20261028-w3-r1`.
Read `launch.json`, `monitor-status.json`, `result.json` and each task's
`cpu-readiness.json`, `generation/native.traj.json`, `generation/model.patch`,
and `model/summary.json`. Raw official reports/logs remain under stage folders.

2026-10-29 image transport recovery: the observed 1ms CDN returned HTTP 200 and
the beginning of the blob for non-zero Range requests, both direct and through
the SSH proxy. Official Docker Hub returned the correct HTTP 206 response for
both large layers. Only `ensure_image` now selects the original repository at
its existing frozen manifest digest, still checking the original configuration
digest. The dedicated daemon uses direct access and one concurrent layer;
the three solve/score slots are unchanged. Seven pre-model image failures are
archived in `infrastructure-history/image-transport-20261029-r1` and restored
to the same task queue. Prompt/model/tool/budget/scoring files are unchanged.
The root launch receipt contains the resumed PID and transport/source hashes.

2026-10-30 precheck recovery: a new run selects only the ten pre-model failures;
the original 19 scored outcomes and the GPU-deferred task remain unchanged.
The user authorized evaluation-only host networking to support official tests
that request remote assets, plus a 60-second CPU per-test limit and one math
library thread. `configs/cpu-evaluation.json` records these settings, shared by
reference, control and model evaluation. Solver configuration remains unchanged
and network-isolated. Lv2 generation additionally passes a fresh zero-API Mini
protocol probe before paid generation. A failed Lv1 gold reference triggers an
independent pristine-source diagnostic; no failing test is skipped or relaxed.
The recovery root is `/root/native-mini-featurebench-lite-precheck10-20261030-w3-r1`.
