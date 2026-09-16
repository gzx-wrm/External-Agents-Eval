"""Post-hoc V4 paired audit; never use verifier-derived output as agent input."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from audit_v3_v1failed28 import ROOT, read, sha

from external_agents_eval.workflows.solve_review_repair.validators import (
    ContractValidationError,
    ContractValidator,
    extract_single_json_object,
)

RESULTS = ROOT / "results/deepswe"
V3 = RESULTS / "deepswe-v1.1-pi-review-repair-v3-v1failed28-w4-20260909"
V4 = RESULTS / "deepswe-v1.1-pi-review-repair-v4-v1failed28-w5-20260909"
VERSIONS = ("p0", "v1", "v3", "v4")


def json_differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                yield {
                    "path": path + "/" + key,
                    "before_present": key in before,
                    "after_present": key in after,
                    "before": before.get(key),
                    "after": after.get(key),
                }
            else:
                yield from json_differences(before[key], after[key], path + "/" + key)
    elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        for index, (old, new) in enumerate(zip(before, after)):
            yield from json_differences(old, new, path + f"/{index}")
    elif before != after:
        yield {"path": path, "before": before, "after": after}


def correction_diagnostic(rr, meta):
    prompt = rr / "reviewer/stage_b_correction_1/prompt.md"
    if meta["final_state"] != "PROTOCOL_VIOLATION" or not prompt.exists():
        return None
    text = prompt.read_text()
    original = json.loads(text.split("ORIGINAL OUTPUT\n", 1)[1])
    trajectory = read(prompt.parent / "trajectory.json")
    corrected = json.loads(extract_single_json_object(trajectory["steps"][-1]["message"])[0])
    validator = ContractValidator(
        ROOT / "external_agents_eval/workflows/solve_review_repair/schemas", contract_version="3.0"
    )
    error = None
    try:
        validator.validate_assessment(
            corrected,
            task_id=meta["task_id"],
            patch_sha256=meta["p0_sha256"],
            ledger_sha256=read(rr / "inputs/assess_patch_input.json")["obligation_ledger"][
                "sha256"
            ],
            ledger=read(rr / "reviewer/review_obligations.json"),
        )
    except ContractValidationError as exc:
        error = str(exc)
    return {
        "initial_errors": json.loads(
            text.split("VALIDATION ERRORS\n", 1)[1].split("ORIGINAL OUTPUT\n", 1)[0]
        ),
        "differences": list(json_differences(original, corrected)),
        "original_verdict": original["verdict"],
        "corrected_verdict": corrected["verdict"],
        "repair_requests": corrected["repair_requests"],
        "offline_schema_semantic_valid": error is None,
        "offline_error": error,
        "repository_checks_replayed": False,
    }


def trials(job):
    found = {}
    for path in job.glob("*/result.json"):
        task = Path(read(path)["config"]["task"]["path"]).name
        assert task not in found, task
        found[task] = path.parent
    return found


def workflow(path):
    rr = path / "review_repair"
    meta = read(rr / "manifest.json")
    assessment = read(rr / "reviewer/patch_assessment.json", {})
    ledger = read(rr / "reviewer/review_obligations.json", {})
    validations = {
        stage: read(rr / f"reviewer/stage_{stage}_validation.json") for stage in ("a", "b")
    }
    trajectory = read(rr / "reviewer/stage_b/trajectory.json", {})
    first_calls = next(
        (s["tool_calls"] for s in trajectory.get("steps", []) if s.get("tool_calls")), []
    )
    ledger_first = any(
        call.get("function_name") == "read"
        and str(call.get("arguments", {}).get("path", "")).endswith("review_obligations.json")
        or call.get("function_name") == "bash"
        and "cat /tmp/external_eval/contracts/review_obligations.json"
        in call.get("arguments", {}).get("command", "")
        for call in first_calls
    )
    return {
        "state": meta["final_state"],
        "verdict": meta.get("reviewer_verdict"),
        "error": meta.get("error"),
        "repair": meta.get("repair_executed"),
        "audit": meta.get("repair_audit_status"),
        "patch_changed": meta["p0_sha256"] != meta["p1_sha256"],
        "obligations": len(ledger.get("obligations", [])),
        "validation": validations,
        "roles": read(rr / "outputs/resource_usage.json", {}).get("roles", []),
        "verdict_rationale": assessment.get("verdict_rationale"),
        "repair_requests": assessment.get("repair_requests", []),
        "source_statuses": dict(
            Counter(item["source_status"] for item in assessment.get("obligation_assessments", []))
        ),
        "implementation_statuses": dict(
            Counter(
                item["implementation_status"]
                for item in assessment.get("obligation_assessments", [])
            )
        ),
        "late_obligations": len(assessment.get("late_obligations", [])),
        "audit_payload": read(rr / "repair/repair_audit.json"),
        "stage_b_first_calls": first_calls,
        "ledger_read_in_first_tool_batch": ledger_first,
        "correction_diagnostic": correction_diagnostic(rr, meta),
        "fingerprints": {
            key: hashlib.sha256(json.dumps(meta.get(key), sort_keys=True).encode()).hexdigest()
            for key in (
                "implementation_sha256",
                "prompt_template_sha256",
                "schema_sha256",
                "few_shot_sha256",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p0 = read(ROOT / "manifests/deepswe_v1.1_pi_random60_p0_20260901.json")
    v1 = read(RESULTS / "deepswe-v1.1-pi-review-repair-v1-complete58/task_index.json")
    v3, v4 = trials(V3), trials(V4)
    expected = {
        task
        for task, item in v1.items()
        if read(Path(item["source_trial_dir"]) / "verifier/reward.json")["reward"] == 0
    }
    assert expected == set(v3) == set(v4) and len(expected) == 28
    rows = []
    for task in sorted(expected):
        paths = {
            "p0": Path(p0["source_job"]["path"]) / p0["tasks"][task]["source_trial_id"],
            "v1": Path(v1[task]["source_trial_dir"]),
            "v3": v3[task],
            "v4": v4[task],
        }
        outcomes = {v: read(path / "result.json") for v, path in paths.items()}
        rewards = {v: read(path / "verifier/reward.json") for v, path in paths.items()}
        test_maps = {
            v: {
                item["name"]: item["status"]
                for item in read(path / "verifier/ctrf.json")["results"]["tests"]
            }
            for v, path in paths.items()
        }
        checks = {
            "task_checksums_equal": len({out["task_checksum"] for out in outcomes.values()}) == 1,
            "test_names_equal": all(set(test_maps[v]) == set(test_maps["p0"]) for v in VERSIONS),
            "rewards_match_results": all(
                rewards[v] == outcomes[v]["verifier_result"]["rewards"] for v in VERSIONS
            ),
            "p0_source_hash_matches": sha(Path(p0["tasks"][task]["patch_path"]))
            == p0["tasks"][task]["patch_sha256"],
        }
        for v in ("v1", "v3", "v4"):
            rr = paths[v] / "review_repair"
            meta = read(rr / "manifest.json")
            checks[f"{v}_p0_matches"] = (
                sha(rr / "inputs/p0.patch")
                == meta["p0_sha256"]
                == p0["tasks"][task]["patch_sha256"]
            )
            checks[f"{v}_submitted_p1_matches"] = (
                sha(rr / "outputs/p1.patch")
                == meta["p1_sha256"]
                == sha(paths[v] / "artifacts/model.patch")
            )
        meta3 = read(paths["v3"] / "review_repair/manifest.json")
        meta4 = read(paths["v4"] / "review_repair/manifest.json")
        checks["v4_versions"] = (
            meta4["prompt_version"] == "4.0"
            and meta4["contract_version"] == "3.0"
            and meta4["experiment_arm"] == "E2"
        )
        checks["schemas_same_as_v3"] = meta3["schema_sha256"] == meta4["schema_sha256"]
        checks["stage_a_repair_corrector_templates_same"] = all(
            meta3["prompt_template_sha256"][name] == meta4["prompt_template_sha256"][name]
            for name in ("build_obligations_v3.md", "repair_once_v3.md", "correct_json_v3.md")
        )
        changes = {}
        for before in ("p0", "v1", "v3"):
            changes[before] = {}
            for suite in ("f2p", "p2p"):
                for direction in ("gained", "lost"):
                    changes[before][f"{direction}_{suite}"] = sorted(
                        name
                        for name, status in test_maps[before].items()
                        if name.startswith(f"[{suite}]")
                        and (
                            status != "passed" and test_maps["v4"][name] == "passed"
                            if direction == "gained"
                            else status == "passed" and test_maps["v4"][name] != "passed"
                        )
                    )
        rows.append(
            {
                "task": task,
                "paths": {v: str(p) for v, p in paths.items()},
                "rewards": rewards,
                "checks": checks,
                "v3": workflow(paths["v3"]),
                "v4": workflow(paths["v4"]),
                "cost_usd": {
                    v: (outcomes[v].get("agent_result") or {}).get("cost_usd") for v in VERSIONS
                },
                "changes_vs": changes,
                "failed_v4_tests": sorted(
                    name for name, status in test_maps["v4"].items() if status != "passed"
                ),
            }
        )
    summary = {"task_count": len(rows), "totals": {}, "workflows": {}}
    for v in VERSIONS:
        totals = {
            key: sum(row["rewards"][v][key] for row in rows)
            for key in ("reward", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total")
        }
        totals.update(
            {
                "macro_f2p": sum(row["rewards"][v]["f2p"] for row in rows) / len(rows),
                "cost_usd": sum(row["cost_usd"][v] or 0 for row in rows),
                "solved": [row["task"] for row in rows if row["rewards"][v]["reward"] == 1],
            }
        )
        summary["totals"][v] = totals
    for v, job in (("v3", V3), ("v4", V4)):
        result = read(job / "result.json")
        config = read(job / "config.json")
        ws = [row[v] for row in rows]
        summary["workflows"][v] = {
            "concurrency": config["n_concurrent_trials"],
            "wall_seconds": (
                datetime.fromisoformat(result["finished_at"])
                - datetime.fromisoformat(result["started_at"])
            ).total_seconds(),
            "states": dict(Counter(w["state"] for w in ws)),
            "verdicts": dict(Counter(w["verdict"] or "no_valid_verdict" for w in ws)),
            "repairs": sum(bool(w["repair"]) for w in ws),
            "audits": dict(Counter(w["audit"] for w in ws)),
            "changed_patches": sum(w["patch_changed"] for w in ws),
            "role_seconds": sum(
                r["resource_usage"].get("wall_time_sec", 0) for w in ws for r in w["roles"]
            ),
            "timeouts": sum(bool(r.get("timed_out")) for w in ws for r in w["roles"]),
            "correction_calls": sum(r["role"] == "json_corrector" for w in ws for r in w["roles"]),
            "fingerprint_variants": {
                k: len({w["fingerprints"][k] for w in ws}) for k in ws[0]["fingerprints"]
            },
            "stage_validation": {
                stage: dict(
                    Counter(
                        "absent"
                        if w["validation"][stage] is None
                        else "valid"
                        if w["validation"][stage]["valid"]
                        else "invalid"
                        for w in ws
                    )
                )
                for stage in ("a", "b")
            },
            "stage_b_trajectory_count": sum(bool(w["stage_b_first_calls"]) for w in ws),
            "ledger_read_in_first_tool_batch": sum(
                w["ledger_read_in_first_tool_batch"] for w in ws
            ),
            "normalization_methods": dict(
                Counter(
                    method
                    for w in ws
                    for validation in w["validation"].values()
                    if validation
                    for method in validation.get("normalization_methods", [])
                )
            ),
        }
    summary["failed_checks"] = {
        row["task"]: [k for k, ok in row["checks"].items() if not ok]
        for row in rows
        if not all(row["checks"].values())
    }
    summary["f2p_directions"] = {
        before: dict(
            Counter(
                "improved"
                if row["rewards"]["v4"]["f2p_passed"] > row["rewards"][before]["f2p_passed"]
                else "declined"
                if row["rewards"]["v4"]["f2p_passed"] < row["rewards"][before]["f2p_passed"]
                else "unchanged"
                for row in rows
            )
        )
        for before in ("p0", "v1", "v3")
    }
    payload = {"purpose": __doc__, "summary": summary, "cases": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
