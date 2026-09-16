# Review note: P002

## Summary

The candidate no longer preserves caller headers on a retry, although the task
authorizes retrying transport failures rather than changing request metadata.
Equivalent base and candidate checks establish a regression in P002.

## Review steps

1. **Bind the existing contract.** I read frozen-base `docs/client.md:20–24`,
   which promises unchanged forwarding of caller headers. The task adds retries
   and does not authorize rewriting those headers.

2. **Compare the same public behavior.** Snapshot roots were supplied by the
   Controller; cwd was `/review-scratch`. I ran the saved probe once per snapshot:

   ```bash
   PYTHONPATH=<snapshot-root> python /review-scratch/probes/retry-headers-v1.py
   ```

   Base printed `attempt2_headers={'X-Request-ID': 'original'}` with exit code 0.
   Candidate printed `attempt2_headers={}` and failed the assertion with exit
   code 1. Both runs used the same input and provisioned dependencies.

3. **Inspect attribution.** Candidate `src/client.py:51–57` clears the header
   mapping before the second transport call. That change explains the observed
   difference and is outside the task-authorized retry behavior.

## Limits and conflicting observations

Only one retry and one header were exercised. No contrary result was observed;
the check does not cover transport-internal headers.

## Reproduction materials

- `probes/retry-headers-v1.py`: executed unchanged on both snapshots.
- `logs/retry-headers-base-v1.txt` and `logs/retry-headers-candidate-v1.txt`:
  captured outputs and exit codes.
