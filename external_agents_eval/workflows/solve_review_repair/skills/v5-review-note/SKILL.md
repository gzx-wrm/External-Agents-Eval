---
name: v5-review-note
description: Write the Markdown review record required for each submitted V5 implementation or regression assessment, using the assigned paths and fixed template.
---

# V5 review note

Read [the template](assets/review-note-template.md) before your first submission
and after context restoration. Use the current Controller `review_notes` object
for paths. Consult an example only when its recording pattern is relevant:

- [Static support](references/static-satisfied.md): code-based evidence, no test run.
- [Dynamic violation](references/dynamic-violated.md): an executed counterexample.
- [Regression violation](references/regression-violated.md): equivalent base and
  candidate checks establish a task-unauthorized break.
- [Blocked investigation](references/blocked-unresolved.md): a failed attempt
  that does not establish the product's behavior.
- [Disputed target](references/source-disputed.md): the proposed requirement is
  not supported by its cited source.

Examples are synthetic writing examples, never evidence for the current task.

For every assessment submitted now, write one note to `output_root` joined with
its assigned `target_paths` entry. Return that exact relative path as
`review_note_path`. Use tools to save and read back the file before final JSON.
All four statuses require notes. Do not create judgments for pending targets,
write another role's records, or overwrite an accepted version. A revision
requires the existing reopen protocol and a newly assigned output location.

Keep the exact title and four headings from the template. Summarize the reviewed
scope and conclusion in one or two sentences. Write numbered review steps as
short paragraphs: purpose, actual action, observed result, and relevance.
For an executed command, record the exact command, actual cwd, snapshot, decisive
output, and exit code only if known. Distinguish executed, blocked, and not run.
For static review, cite inspected code and explain the implication; do not invent
commands or run tests merely to fill the template. For a regression, explain the
reference behavior, candidate difference, and why the task did not authorize it.

Use `disputed` only when the target's source or applicability is unsupported,
ambiguous, or out of scope. Use `unresolved` when the target is applicable but
the available evidence cannot settle its behavior. Both require the same note
template and an explicit account of what remains uncertain.

Preserve material contrary observations and limitations. Keep relevant details
from earlier turns, identifying their origin when known; do not invent event IDs.
List existing probes, inputs, and logs under the supplied `probe_root`/`log_root`.
When changing an executed probe, save a new version. State missing materials or
uncertain execution-time versions; saved files alone do not establish replay.

Keep the note and JSON consistent in target, status, expectation, observation,
and uncertainty. The note records observable actions and concise explanations,
not an internal thought transcript or a new requirement source. Aim for 120–250
English words excluding commands and short output excerpts; this is a soft limit.
Do not copy complete logs, files, the ledger, or unrelated investigation history.

Save every note for the current batch before responding. On a write failure,
correct it within the existing budget; never pretend the file exists. Return
only ReviewBatch JSON in one `json` code block, with no Markdown note pasted into
the final response. Controller validation determines whether the batch commits.
