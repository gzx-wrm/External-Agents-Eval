"""Post-hoc failure audit. Never use its verifier evidence as agent input."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/deepswe"
JOBS = [
    "deepswe-v1.1-pi-review-repair-primary58-v1",
    "deepswe-v1.1-pi-review-repair-canary3-v13",
    "deepswe-v1.1-pi-review-repair-recovery11-v13",
]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", default="")
    parser.add_argument("--view", choices=["full", "failures", "review"], default="full")
    args = parser.parse_args()
    baseline = read(ROOT / "manifests/deepswe_v1.1_pi_random60_p0_20260901.json")
    selected = {}
    for job in JOBS:
        for path in sorted((RESULTS / job).glob("*/result.json")):
            trial = read(path)
            task = Path(trial["config"]["task"]["path"]).name
            if task in baseline["metric_task_ids"]:
                selected[task] = (path.parent, trial)
    assert set(selected) == set(baseline["metric_task_ids"])
    rows = []
    for task in baseline["metric_task_ids"]:
        directory, trial = selected[task]
        workflow = directory / "review_repair"
        manifest = read(workflow / "manifest.json")
        p0 = read(
            Path(baseline["source_job"]["path"])
            / baseline["tasks"][task]["source_trial_id"]
            / "result.json"
        )
        p0_rewards = p0["verifier_result"]["rewards"]
        p1_rewards = read(directory / "verifier/reward.json")
        assert p1_rewards == trial["verifier_result"]["rewards"], task
        assert manifest["p0_sha256"] == baseline["tasks"][task]["patch_sha256"], task
        assert sha(workflow / "inputs/p0.patch") == manifest["p0_sha256"], task
        assert sha(workflow / "outputs/p1.patch") == manifest["p1_sha256"], task
        assert sha(directory / "artifacts/model.patch") == manifest["p1_sha256"], task
        if p1_rewards["reward"] == 1:
            continue
        ctrf = read(directory / "verifier/ctrf.json", {})
        tests = ctrf.get("results", {}).get("tests", [])
        failed = [t for t in tests if t.get("status") == "failed"]
        events = [json.loads(line) for line in (workflow / "events.jsonl").read_text().splitlines()]
        assessment = read(workflow / "reviewer/patch_assessment.json", {})
        ledger = read(workflow / "reviewer/review_obligations.json", {})
        roles = read(workflow / "outputs/resource_usage.json", {}).get("roles", [])
        row = {
            "task": task,
            "trial_dir": str(directory),
            "source_job": directory.parent.name,
            "p0": p0_rewards,
            "p1": p1_rewards,
            "state": manifest["final_state"],
            "error": manifest.get("error"),
            "verdict": manifest.get("reviewer_verdict"),
            "repair_executed": manifest.get("repair_executed"),
            "patch_changed": manifest["p0_sha256"] != manifest["p1_sha256"],
            "patch_bytes": {
                "p0": (workflow / "inputs/p0.patch").stat().st_size,
                "p1": (workflow / "outputs/p1.patch").stat().st_size,
            },
            "failed_tests": failed,
            "test_statuses": dict(Counter(t.get("status") for t in tests)),
            "ledger": ledger,
            "assessment": assessment,
            "events": events,
            "roles": roles,
        }
        rows.append(row)
    assert len(rows) == 28
    report = {
        "purpose": "Post-hoc audit only; contains held-out verifier evidence. Do not feed to agents.",
        "selection": JOBS,
        "evaluated": 58,
        "failed": len(rows),
        "verified": "All 58 source P0, workflow P1 and submitted artifact hashes agree.",
        "cases": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        if args.task and args.task not in row["task"]:
            continue
        if args.view == "full":
            print(json.dumps(row, ensure_ascii=False))
            continue
        print(
            "\nTASK",
            row["task"],
            "F2P",
            f"{row['p0']['f2p_passed']}->{row['p1']['f2p_passed']}/{row['p1']['f2p_total']}",
            "state",
            row["state"],
            "verdict",
            row["verdict"],
            "changed",
            row["patch_changed"],
        )
        if args.view == "failures":
            for test in row["failed_tests"]:
                print(test["name"], "\n", test.get("message", "")[:1700])
        else:
            print("RATIONALE", row["assessment"].get("verdict_rationale"))
            for request in row["assessment"].get("repair_requests", []):
                print("REQUEST", json.dumps(request, ensure_ascii=False))
            if row["error"]:
                print("ERROR", row["error"])


if __name__ == "__main__":
    main()
