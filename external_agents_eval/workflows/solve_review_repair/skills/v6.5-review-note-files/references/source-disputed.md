# Review note: O003

## Summary

O003 is disputed because its cited task sentence requires stable output but does
not require the proposed alphabetical ordering. The candidate's behavior is
observable, but this target overstates the contract.

## Review steps

1. **Check the target against its source.** O003 says results must be sorted
   alphabetically. The cited task text says only “return results in a stable
   order.” Stability constrains repeatability; it does not select alphabetical
   order, so the stronger expectation has no source.

2. **Inspect relevant public context.** Frozen-base `docs/results.md:12–17`
   also promises deterministic output without naming a sort key or direction.
   I found no applicable public contract supporting the added alphabetical rule.

3. **Separate applicability from implementation.** Candidate
   `src/results.py:30–38` preserves insertion order. I did not run a probe because
   observing a different order could not establish a violation of an unsupported
   alphabetical requirement.

## Limits and conflicting observations

The broader stability requirement may still need a separate assessment. This
note disputes only O003's unsupported ordering detail and does not establish that
the candidate is stable for all inputs.

## Reproduction materials

Static review only. No probe or log was produced; recheck the cited task text,
base documentation, and candidate range in their Controller-bound snapshots.
