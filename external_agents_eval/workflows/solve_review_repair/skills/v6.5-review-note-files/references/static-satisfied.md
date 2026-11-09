# Review note: P001

## Summary

The inspected candidate request path preserves caller-provided headers when
forwarding to the transport. P001 is satisfied within this static review scope.

## Review steps

1. **Confirm the preservation contract.** I read base `docs/client.md:20–24`,
   which promises unchanged forwarding of caller-provided headers. The task adds
   retries and does not authorize changing this contract.

2. **Follow the forwarding path.** I inspected base and candidate
   `src/client.py:40–58`. In both snapshots, the public method passes `headers`
   directly to the transport without an intervening mutation. In the candidate,
   each retry uses that same argument. This supports the specified forwarding
   behavior in the inspected path.

## Limits and conflicting observations

No runtime check was executed. This review does not establish transport-internal
behavior or behavior of external subclasses. No contrary observation was found
in the inspected path.

## Reproduction materials

Static review only; no probe or output log was produced. Reinspect the cited
ranges in the Controller-bound base and candidate snapshots.
