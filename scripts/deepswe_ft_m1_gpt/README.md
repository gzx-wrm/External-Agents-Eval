# FT ledger + native Mini + GPT-6.1 Sol

This independent model comparison reuses the original FT pre-Solver ledger and
the exact augmented task text from `ft_m1`. It starts Solver from the original
repository image. It does not reuse an M1 patch, conversation, evidence packet,
review guidance or subsequent reviewer ledger additions.

The five selected cases are the final reward=0 cases from `ft_m1_review`:
LangChain request coalescing, Termenv ANSI resets, Vulture cache, Obsidian TOC,
and GQL incremental delivery. This is a failure-selected recovery pilot, not a
new full-suite pass@1 estimate.

`prepare` freezes task, ledger, image, source and original outcome identities.
The corrected M0 generation/isolation/grading helpers and ledger injection are
copied byte for byte into the independent bundle. Original runs remain intact.
Native Mini 2.4.6 system/task/observation/error templates, Bash, Git sanitization,
offline containers, 2 CPUs/8GiB, 3-hour solve budget, 65536 output tokens, no
step/cost stop, and immediate original DeepSWE scoring are retained. Three
workers each solve and then score before taking another case.

The model adapter extends Mini's native Responses model. The two supplied
gateways support streamed Responses; one rejects Chat Completions on its Codex
channel. Explicitly passing the unchanged Mini system message as `instructions`
avoids the other gateway's default instructions. No system message is duplicated
inside `input`. The native Bash schema, result formatting, action parsing and
full response/history serialization remain in use. Raw completed stream events
are retained rather than SDK parsed-output helper objects.
On outbound reasoning items only, omit the SDK's output status and null
placeholders rejected by these gateways. Raw trajectory items, meaningful
reasoning content, summaries and encrypted content are unchanged.

Initial endpoints alternate by task. Conversations stay on their current
endpoint for cache reuse. A transport, authentication or service failure can
retry the identical request at the other endpoint; no Bash command is replayed.
All history, including opaque reasoning items, is retained. Malformed requests
remain explicit errors; history is never silently dropped to make failover work.
The native model retry loop remains; the SDK's additional retries are disabled.

Credentials live in an owner-only host file outside the experiment directory.
They are read only when sending a request and never placed in the serializable
model configuration or forwarded into task containers. Per-task route JSONL
records endpoint, model, request/history fingerprints, status and raw token usage,
without credentials or new model-visible content.

The preflight uses the real native agent and original Docker task with simulated
model responses to verify the full task/ledger → Responses → Bash → observation
→ submission → patch export path. It makes zero model calls. Prior M0 official
gold/control verification is explicitly referenced, not claimed as newly run.
Separate live gateway probes test connectivity and multi-turn tool history, not
benchmark solving. Reasoning effort and pipeline scope require human confirmation
before the paid five-case launch.

Official model reference:
https://developers.openai.com/api/docs/models/gpt-6.1-sol

For the authorized complementary run, `prepare_remaining.py` verifies the same
fixed FT30, excludes the five tasks with effective official GPT results, and
freezes the remaining 25 original M1 inputs. It copies all six corrected runtime
scripts byte for byte. Existing `preflight` and `launch.py` entry points then
run the new batch with the same settings. The two batches must be merged by
unique task ID, using the effective five-case results rather than initial
protocol-failed attempts; neither those attempts nor connectivity probes are
additional benchmark samples.
