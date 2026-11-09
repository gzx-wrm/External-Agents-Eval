# Review note: O001

## Summary

The candidate public entry point reversed a two-element input despite the task's
order-preservation requirement. The observed counterexample violates O001.

## Review steps

1. **Check the expected behavior.** I matched O001 to the task's explicit
   requirement to preserve input order. The expectation comes from that text,
   not from the probe or the candidate implementation.

2. **Exercise the public entry point.** I ran the saved probe with input
   `['first', 'second']`. It calls the public API and compares the actual sequence
   with the unchanged input. Snapshot: candidate; cwd: `/work/candidate`.

   ```bash
   python /review-scratch/probes/order-v1.py
   ```

   Decisive output: `observed=['second', 'first']`, followed by an assertion
   failure against `['first', 'second']`. The tool reported exit code 1. The
   mismatch is a behavioral counterexample, not a failure to start the probe.

## Limits and conflicting observations

Only this two-element input was exercised; no second replay was performed.
This establishes the ordering violation but does not identify its root cause.

## Reproduction materials

- `/review-scratch/probes/order-v1.py`: the executed probe; no later edits.
- `/review-scratch/logs/order-v1.txt`: the captured output.

Replay requires the recorded candidate snapshot and its provisioned environment.
