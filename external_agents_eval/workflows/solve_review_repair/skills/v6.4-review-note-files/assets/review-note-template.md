# Review note: {{TARGET_ID}}

## Assessment

```json
{"target_id": "{{TARGET_ID}}", "status": "{{satisfied|violated|unresolved|disputed}}", "expected": "{{...}}", "observed": "{{...}}", "reason": "{{...}}", "code_refs": []}
```

## Summary

{{One or two sentences describing the reviewed scope and conclusion.}}

## Review steps

1. **{{Purpose of this step}}**

   {{Describe what was actually inspected or executed, what was observed,
   and why it matters. Cite relevant snapshot/path/line ranges.}}

   {{For an executed command only: actual cwd and base/candidate snapshot.}}

   ```bash
   {{Exact command actually executed; omit this block for a static-only step.}}
   ```

   {{Decisive observed output and known exit code, or state it was not recorded.}}

{{Add only the additional steps needed to support this assessment.}}

## Limits and conflicting observations

{{State unverified scope, blocking conditions, contrary results, and any change
in probe inputs or expectations. Do not convert missing coverage into success.}}

## Reproduction materials

{{List existing probe/input/log paths, purpose, and replay conditions. State
missing materials or uncertain versions; for static-only review say so.}}
