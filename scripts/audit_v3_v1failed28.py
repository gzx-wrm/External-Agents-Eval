"""Post-hoc paired audit of V3's 28-task run; never provide its output to agents."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/deepswe"
JOB = RESULTS / "deepswe-v1.1-pi-review-repair-v3-v1failed28-w4-20260909"


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frozen = read(ROOT / "manifests/deepswe_v1.1_pi_random60_p0_20260901.json")
    index = read(RESULTS / "deepswe-v1.1-pi-review-repair-v1-complete58/task_index.json")
    run = read(JOB.with_name(JOB.name + ".run.json"))
    job_result = read(JOB / "result.json")
    expected = {
        task
        for task, item in index.items()
        if read(Path(item["source_trial_dir"]) / "verifier/reward.json")["reward"] == 0
    }
    assert expected == set(run["task_ids"]) and len(expected) == 28
    trials = {}
    for path in JOB.glob("*/result.json"):
        result = read(path)
        task = Path(result["config"]["task"]["path"]).name
        assert task not in trials
        trials[task] = path.parent
    assert set(trials) == expected
    rows = []
    for task in sorted(expected):
        directories = {
            "p0": Path(frozen["source_job"]["path"]) / frozen["tasks"][task]["source_trial_id"],
            "v1": Path(index[task]["source_trial_dir"]),
            "v3": trials[task],
        }
        results = {version: read(path / "result.json") for version, path in directories.items()}
        rewards = {
            version: read(path / "verifier/reward.json") for version, path in directories.items()
        }
        tests = {
            version: {
                test["name"]: test["status"]
                for test in read(path / "verifier/ctrf.json", {})
                .get("results", {})
                .get("tests", [])
            }
            for version, path in directories.items()
        }
        rr = trials[task] / "review_repair"
        manifest = read(rr / "manifest.json")
        ledger = read(rr / "reviewer/review_obligations.json", {})
        assessment = read(rr / "reviewer/patch_assessment.json", {})
        validations = {
            stage: read(rr / f"reviewer/stage_{stage}_validation.json") for stage in ("a", "b")
        }
        roles = read(rr / "outputs/resource_usage.json", {}).get("roles", [])
        checks = {
            "task_checksum_matches": len({x["task_checksum"] for x in results.values()}) == 1,
            "test_names_match": set(tests["p0"]) == set(tests["v1"]) == set(tests["v3"]),
            "reward_matches_result": all(
                rewards[v] == results[v]["verifier_result"]["rewards"] for v in rewards
            ),
            "source_p0_hash_matches": sha(Path(frozen["tasks"][task]["patch_path"]))
            == frozen["tasks"][task]["patch_sha256"],
        }
        for version in ("v1", "v3"):
            workflow = directories[version] / "review_repair"
            meta = read(workflow / "manifest.json")
            checks[f"{version}_p0_hash_matches"] = (
                sha(workflow / "inputs/p0.patch")
                == meta["p0_sha256"]
                == frozen["tasks"][task]["patch_sha256"]
            )
            checks[f"{version}_submitted_p1_matches"] = (
                sha(workflow / "outputs/p1.patch")
                == meta["p1_sha256"]
                == sha(directories[version] / "artifacts/model.patch")
            )

        def changed_tests(before, after, prefix, gained, *, tests=tests):
            return sorted(
                name
                for name in set(tests[before]) & set(tests[after])
                if name.startswith(prefix)
                and (
                    (tests[before][name] != "passed" and tests[after][name] == "passed")
                    if gained
                    else (tests[before][name] == "passed" and tests[after][name] != "passed")
                )
            )

        rows.append(
            {
                "task": task,
                "directories": {v: str(p) for v, p in directories.items()},
                "rewards": rewards,
                "checks": checks,
                "state": manifest["final_state"],
                "error": manifest.get("error"),
                "verdict": manifest.get("reviewer_verdict"),
                "repair_executed": manifest.get("repair_executed"),
                "audit_status": manifest.get("repair_audit_status"),
                "patch_changed_from_p0": manifest["p0_sha256"] != manifest["p1_sha256"],
                "stage_validation": validations,
                "roles": roles,
                "cost_usd": {
                    v: (x.get("agent_result") or {}).get("cost_usd", 0) for v, x in results.items()
                },
                "obligations": len(ledger.get("obligations", [])),
                "source_statuses": dict(
                    Counter(
                        x["source_status"] for x in assessment.get("obligation_assessments", [])
                    )
                ),
                "implementation_statuses": dict(
                    Counter(
                        x["implementation_status"]
                        for x in assessment.get("obligation_assessments", [])
                    )
                ),
                "verdict_rationale": assessment.get("verdict_rationale"),
                "repair_requests": assessment.get("repair_requests", []),
                "gained_f2p_vs_v1": changed_tests("v1", "v3", "[f2p]", True),
                "lost_f2p_vs_v1": changed_tests("v1", "v3", "[f2p]", False),
                "gained_f2p_vs_p0": changed_tests("p0", "v3", "[f2p]", True),
                "lost_f2p_vs_p0": changed_tests("p0", "v3", "[f2p]", False),
                "gained_p2p_vs_p0": changed_tests("p0", "v3", "[p2p]", True),
                "lost_p2p_vs_p0": changed_tests("p0", "v3", "[p2p]", False),
                "failed_test_names": sorted(
                    name for name, status in tests["v3"].items() if status == "failed"
                ),
                "fingerprints": {
                    key: hashlib.sha256(
                        json.dumps(manifest.get(key), sort_keys=True).encode()
                    ).hexdigest()
                    for key in (
                        "implementation_sha256",
                        "prompt_template_sha256",
                        "schema_sha256",
                        "few_shot_sha256",
                    )
                },
            }
        )
    totals = {}
    for version in ("p0", "v1", "v3"):
        totals[version] = {
            field: sum(row["rewards"][version][field] for row in rows)
            for field in ("reward", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total")
        }
        totals[version]["cost_usd"] = sum(row["cost_usd"][version] or 0 for row in rows)
        totals[version]["macro_f2p"] = sum(row["rewards"][version]["f2p"] for row in rows) / len(
            rows
        )
    summary = {
        "selected_count": 28,
        "started_at": job_result["started_at"],
        "finished_at": job_result["finished_at"],
        "wall_time_sec": (
            datetime.fromisoformat(job_result["finished_at"])
            - datetime.fromisoformat(job_result["started_at"])
        ).total_seconds(),
        "job_stats": {k: v for k, v in job_result["stats"].items() if k != "evals"},
        "totals": totals,
        "states": dict(Counter(row["state"] for row in rows)),
        "verdicts": dict(Counter(row["verdict"] or "no_valid_verdict" for row in rows)),
        "audits": dict(Counter(row["audit_status"] for row in rows)),
        "patch_changed_count": sum(row["patch_changed_from_p0"] for row in rows),
        "f2p_task_direction_vs_v1": dict(
            Counter(
                "improved"
                if row["rewards"]["v3"]["f2p_passed"] > row["rewards"]["v1"]["f2p_passed"]
                else "declined"
                if row["rewards"]["v3"]["f2p_passed"] < row["rewards"]["v1"]["f2p_passed"]
                else "unchanged"
                for row in rows
            )
        ),
        "failed_checks": {
            row["task"]: [k for k, v in row["checks"].items() if not v]
            for row in rows
            if not all(row["checks"].values())
        },
        "fingerprint_variant_counts": {
            k: len({row["fingerprints"][k] for row in rows}) for k in rows[0]["fingerprints"]
        },
        "role_timeout_count": sum(
            role.get("timed_out", False) for row in rows for role in row["roles"]
        ),
    }
    payload = {
        "purpose": "Post-hoc verifier analysis only. Never provide this artifact to agents.",
        "summary": summary,
        "cases": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
