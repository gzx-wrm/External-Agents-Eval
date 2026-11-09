# Review note: X002

## Summary

X002 claims the parser accepts a bracketed binding after the new keyword as an
ordinary expression. The task text requires every destructuring pattern in the new
declaration to be rejected, so the literal reading is supported and X002 is violated.

## Review steps

1. **State the readings.** Reading A (claimed candidate behavior): `kw [a] = b;` parses
   as an expression statement. Reading B (literal task reading): it is a declaration
   with a destructuring pattern and must fail with the documented error substring.

2. **Run a discriminating probe.** The saved probe parses `kw [a] = b;` and `kw {a} = b;`
   through the public `parse` entry point with the feature option enabled.
   Snapshot: candidate; cwd: candidate_root.

   ```bash
   node /review-scratch/probes/x002-v1.mjs
   ```

   Decisive output: `[a]` produced an ExpressionStatement; `{a}` threw the documented
   error. Exit code 0. The candidate implements reading A.

3. **Weigh the evidence.** Class (a): the task states that a destructuring pattern in
   the declaration must fail; it does not exempt bracketed patterns. Class (c): a
   language proposal treats the bracketed form as a member expression, but general
   knowledge does not override the task text. Reading B is supported.

4. **Register the derived obligation.** New target: "With the feature option enabled,
   `kw [a] = b;` fails with an error message containing the documented substring",
   citing the task sentence. It is assessed in the next round with this probe.

## Limits and conflicting observations

Only module source was probed. The class (c) evidence conflicts with the task and was
given no weight. No base-repository convention addresses the bracketed form.

## Reproduction materials

- `/review-scratch/probes/x002-v1.mjs`: the executed probe; no later edits.
- `/review-scratch/logs/x002-v1.txt`: the captured output.

Replay requires the recorded candidate snapshot and its provisioned environment.
