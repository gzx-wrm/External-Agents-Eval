"""V6.8 end-to-end repair test (no Solver; user, 2026-11-08).

For each failing candidate of the patch-review experiment, v68_repair.select_issues builds the issue
list: review defect AND the C1 resolver's reading differs from the candidate's (at most
v68_repair.MAX_ISSUES). Inputs are the review rows, the candidate reading mapping and the C1
resolutions; no reference solution or hidden test. Phases:
- control (zero API): a fresh M0-prepared offline baseline container with the candidate applied;
  snapshot against the synthetic baseline commit exactly as M0 exports model.patch; graded by the
  unchanged official verifier in the frozen task image.
- run: the same container setup plus one native Mini repair episode (M0's model and templates; only
  the task section and limits differ). The snapshot after the episode is graded the same way.
Both snapshots pass through the v6.6.4 submission filter (newly added test files removed) before
grading. prepare and preflight make no model calls; outputs go to a new root; a finished episode or
grade is never resampled.

usage: repair.py prepare --root NEW --review-root R --map-root M --resolve-root C --note TEXT
       repair.py preflight --root ROOT
       repair.py control --root ROOT [--containers N] [--graders N]
       repair.py run --root ROOT [--containers N] [--graders N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.benchmarks.deepswe.submission_filter import filter_record, strip_added_tests  # noqa: E402
from external_agents_eval.workflows.solve_review_repair import v68_repair as rp  # noqa: E402
from external_agents_eval.workflows.solve_review_repair import v68_repair_lint as plint  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rev = _load("v68_repair_review", HERE / "review.py")
base, read, save, sha, pair_key = rev.base, rev.read, rev.save, rev.sha, rev.pair_key

EXPERIMENT_ID = "v6.8-repair"
RUNNER_VERSION = "v6.8-repair-1"
CONFIG = "repair-mini.yaml"
FROZEN_SOURCES = (
    "scripts/deepswe_v68_gen/repair.py",
    "scripts/deepswe_v68_gen/review.py",
    "scripts/deepswe_v68_gen/run.py",
    "external_agents_eval/workflows/solve_review_repair/v68_repair.py",
    "external_agents_eval/workflows/solve_review_repair/v68_repair_lint.py",
    "external_agents_eval/benchmarks/deepswe/submission_filter.py",
)
LINT_PATH = PROJECT / "external_agents_eval/workflows/solve_review_repair/v68_repair_lint.py"
APPLY = f"cd /app && git apply --whitespace=nowarn {rp.WORK_DIR}/candidate.diff"


def code_hashes():
    return {name: sha(PROJECT / name) for name in FROZEN_SOURCES}


def prepare(root, review_root, map_root, resolve_root, note):
    import yaml

    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    manifest = read(resolve_root / "manifest.json")
    if manifest.get("prompt") != "evidence" or manifest.get("order") != "original":
        raise ValueError("The resolve root must be the direction C1 run (prompt evidence, original order)")
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    for target, source in base.NATIVE_COPIES.items():
        digests = {sha(Path(m) / "scripts" / source) for m in base.M0_ROOTS}
        if len(digests) != 1:
            raise ValueError("M0 roots disagree on " + source)
        shutil.copyfile(Path(base.M0_ROOTS[0]) / "scripts" / source, root / "scripts" / target)
    native = base.load_native(root)
    (root / "configs").mkdir()
    config = native.prepare_config(root)
    for m0 in base.M0_ROOTS:
        if config != yaml.safe_load((Path(m0) / "configs/native-deepseek-v4.1-flash.yaml").read_text()):
            raise ValueError("Mini configuration differs from M0: " + m0)
    (root / "configs" / CONFIG).write_text(yaml.safe_dump(
        base.episode_config(config, rp.header(), rp.STEP_LIMIT, rp.WALL_TIME_SEC), allow_unicode=True,
        sort_keys=False))
    (root / "patches").mkdir()
    (root / "issues").mkdir()
    rows = []
    for source in read(review_root / "selection.json")["pairs"]:
        key = pair_key(source)
        if sha(source["patch"]) != source["patch_sha256"]:
            raise ValueError(f"{key}: review patch changed")
        patch_path = root / "patches" / f"{key}.diff"
        shutil.copyfile(source["patch"], patch_path)
        review_file = review_root / "pairs" / key / "review.json"
        review_rows = {r["id"]: r for r in read(review_file)["rows"]} if review_file.exists() else {}
        mapping = read(map_root / key / "mapping.json") if (map_root / key / "mapping.json").exists() else None
        candidate = mapping["labels"] if mapping and mapping["ok"] else {}
        answers = read(resolve_root / source["task_id"] / "resolution.json")["answers"]
        issues = rp.select_issues(read(source["data"]), review_rows, candidate, answers)
        issue_path = root / "issues" / f"{key}.json"
        save(issue_path, {"pair": key, "issues": issues})
        (root / "issues" / f"{key}.md").write_text(rp.issues_markdown(issues))
        recorded = read(Path(source["candidate_dir"]) / "result.json")
        rows.append({"task_id": source["task_id"], "candidate": source["candidate"],
                     "candidate_dir": source["candidate_dir"], "patch": str(patch_path), "patch_sha256": sha(patch_path),
                     "issues": str(issue_path), "issues_sha256": sha(issue_path), "issue_count": len(issues),
                     "recorded_reward": recorded.get("reward"),
                     "recorded_official": recorded.get("official"), "task_row": source["task_row"]})
    selection = {"experiment_id": EXPERIMENT_ID, "version": rp.VERSION, "runner_version": RUNNER_VERSION,
                 "note": note, "pair_count": len(rows), "pairs": rows,
                 "inputs": {"review_root": str(review_root), "map_root": str(map_root),
                            "resolve_root": str(resolve_root), "resolve_manifest": manifest},
                 "code_sha256": code_hashes(),
                 "native_source_sha256": {n: sha(root / "scripts" / n) for n in base.NATIVE_COPIES},
                 "config_sha256": {CONFIG: sha(root / "configs" / CONFIG)},
                 "limits": {"repair": [rp.STEP_LIMIT, rp.WALL_TIME_SEC], "max_issues": rp.MAX_ISSUES}}
    save(root / "selection.json", selection)
    return selection


def validate(root):
    import yaml

    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID or code_hashes() != selection["code_sha256"]:
        raise ValueError("Wrong experiment or frozen code changed")
    for name, digest in selection["native_source_sha256"].items():
        if sha(root / "scripts" / name) != digest:
            raise ValueError("Frozen native source changed: " + name)
    if sha(root / "configs" / CONFIG) != selection["config_sha256"][CONFIG]:
        raise ValueError("Episode configuration changed")
    for row in selection["pairs"]:
        if sha(row["patch"]) != row["patch_sha256"] or sha(row["issues"]) != row["issues_sha256"]:
            raise ValueError(f"{pair_key(row)}: frozen input changed")
    return selection, yaml.safe_load((root / "configs" / CONFIG).read_text())


def snapshot(native, environment, container_dir):
    commit = read(container_dir / "isolation.json")["synthetic_commit"]
    return native.checked(environment, native.snapshot_command(commit), timeout=600)


def episode(native, root, row, config, output, *, agent_on, model_patch=None):
    """Fresh baseline container with the candidate applied; optionally one repair episode."""
    if (output / "episode.json").exists():
        return read(output / "episode.json")
    if output.exists():
        raise RuntimeError(f"{output} has a partial episode; not resampled")
    issues = read(row["issues"])["issues"]
    record = {"task_id": row["task_id"], "candidate": row["candidate"], "agent": agent_on, "started_at": native.now()}
    environment = None
    try:
        environment, container = base.baseline_environment(native, root, row["task_row"], config, output / "container")
        files = {"task.md": row["task_row"]["task"]["instruction"].encode(), "candidate.diff": Path(row["patch"]).read_bytes(),
                 "issues.md": rp.issues_markdown(issues).encode(), "scope.json": rp.scope(issues),
                 "lint.py": LINT_PATH.read_bytes()}
        for name, data in files.items():
            native.upload(container, data, f"{rp.WORK_DIR}/{name}")
        applied = environment.execute({"command": APPLY}, timeout=300)
        record["patch_applied"] = applied["returncode"] == 0
        (output / "apply.txt").write_text(applied["output"])
        if not record["patch_applied"]:
            record["exit_status"] = "patch_failed"
        else:
            pre = snapshot(native, environment, output / "container")
            (output / "pre.patch").write_text(pre)
            if agent_on:
                agent = native.DefaultAgent(native.get_model(config=config["model"]), environment, **config["agent"],
                                            output_path=output / "traj.json")
                try:
                    if model_patch is not None:
                        with patch("litellm.completion", model_patch):
                            info = agent.run(row["task_row"]["task"]["instruction"])
                    else:
                        info = agent.run(row["task_row"]["task"]["instruction"])
                    record["exit_status"] = info.get("exit_status")
                except Exception as error:  # noqa: BLE001 - keep what the agent changed
                    record.update(exit_status="agent_error", error_type=type(error).__name__, error=str(error)[:2000])
                record["steps"] = sum(1 for m in getattr(agent, "messages", [])
                                      if isinstance(m, dict) and m.get("role") == "assistant")
                lint = environment.execute({"command": f"cd {rp.WORK_DIR} && python lint.py"}, timeout=120)
                record["lint_ok"] = lint["returncode"] == 0
                (output / "lint.txt").write_text(lint["output"])
                chunks, _ = container.get_archive(rp.WORK_DIR)
                base.untar_json(b"".join(chunks), output / "files")
                post = snapshot(native, environment, output / "container")
                (output / "post.patch").write_text(post)
                record["code_changed"] = post != pre
    finally:
        base.close(native, environment)
    record["finished_at"] = native.now()
    save(output / "episode.json", record)
    return record


def grade(native, row, snapshot_text, output, graders):
    if (output / "grade.json").exists():
        return read(output / "grade.json")
    submitted, removed = strip_added_tests(snapshot_text)
    output.mkdir(parents=True, exist_ok=True)
    (output / "submitted.patch").write_text(submitted)
    save(output / "submission-filter.json", filter_record(snapshot_text, submitted, removed))
    task_row = row["task_row"]
    with graders:
        result = native.grade(Path(task_row["m0_root"]), task_row["task"], task_row["image"]["id"], submitted,
                              output / "evaluation")
    result["removed_test_files"] = removed
    save(output / "grade.json", result)
    return result


def _counts(official):
    return {k: (official or {}).get(k) for k in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")}


def run_control(native, root, row, config, containers, graders):
    out = root / "pairs" / pair_key(row) / "control"
    if (out / "result.json").exists():
        return read(out / "result.json")
    with containers:
        record = episode(native, root, row, config, out / "episode", agent_on=False)
    result = {"pair": pair_key(row), "phase": "control", "patch_applied": record["patch_applied"],
              "recorded_reward": row["recorded_reward"], "recorded": _counts(row["recorded_official"])}
    if record["patch_applied"]:
        graded = grade(native, row, (out / "episode" / "pre.patch").read_text(), out / "grade", graders)
        result.update(status=graded["status"], reward=graded.get("reward"), counts=_counts(graded.get("official")),
                      removed_test_files=graded["removed_test_files"])
    else:
        result["status"] = "patch_failed"
    save(out / "result.json", result)
    return result


def run_repair(native, root, row, config, containers, graders, *, model_patch=None, grade_on=True):
    out = root / "pairs" / pair_key(row) / "repair"
    if (out / "result.json").exists():
        return read(out / "result.json")
    result = {"pair": pair_key(row), "phase": "repair", "issues": row["issue_count"]}
    if row["issue_count"] == 0:
        result["status"] = "no_issues"
        save(out / "result.json", result)
        return result
    with containers:
        record = episode(native, root, row, config, out / "episode", agent_on=True, model_patch=model_patch)
    files = out / "episode" / "files"
    decisions = []
    if (files / "scope.json").exists():
        ids, rows, _ = plint.load(str(files))
        decisions = [{k: v for k, v in r.items() if k != "_line"} for r in plint.final_rows(ids, rows)]
    result.update({k: record.get(k) for k in ("patch_applied", "exit_status", "steps", "lint_ok", "code_changed")})
    result["decisions"] = {"changed": sum(d.get("decision") == "changed" for d in decisions),
                           "kept": sum(d.get("decision") == "kept" for d in decisions), "rows": decisions}
    if grade_on and record["patch_applied"]:
        graded = grade(native, row, (out / "episode" / "post.patch").read_text(), out / "grade", graders)
        result.update(status=graded["status"], reward=graded.get("reward"), counts=_counts(graded.get("official")),
                      removed_test_files=graded["removed_test_files"])
    else:
        result["status"] = "not_graded" if record["patch_applied"] else "patch_failed"
    save(out / "result.json", result)
    return result


def run_all(root, selection, config, phase, max_containers, max_graders, *, model_patch=None, grade_on=True,
            only=None):
    native = base.load_native(root)
    containers = threading.BoundedSemaphore(max_containers)
    graders = threading.BoundedSemaphore(max_graders)
    rows = [r for r in selection["pairs"] if not only or pair_key(r) in only]
    lock, results = threading.Lock(), {}

    def one(row):
        try:
            base.ensure_image(native, row["task_row"])
            if phase == "control":
                result = run_control(native, root, row, config, containers, graders)
            else:
                result = run_repair(native, root, row, config, containers, graders, model_patch=model_patch,
                                    grade_on=grade_on)
        except Exception as error:  # noqa: BLE001 - keep per-pair failures explicit
            result = {"pair": pair_key(row), "phase": phase, "status": "stage_error",
                      "error_type": type(error).__name__, "error": str(error)[:2000]}
            save(root / "errors" / phase / f"{pair_key(row)}.json", result)
        with lock:
            results[pair_key(row)] = result
            save(root / f"result-{phase}.json", {"experiment_id": EXPERIMENT_ID, "results": list(results.values())})
        print(json.dumps({k: result.get(k) for k in ("pair", "phase", "status", "reward", "counts", "issues")}),
              flush=True)

    with ThreadPoolExecutor(max_workers=max(1, len(rows))) as pool:
        for future in as_completed([pool.submit(one, row) for row in rows]):
            future.result()
    return results


def preflight(root, selection, config):
    """Zero API: a stub agent records 'kept' for every issue and submits; checks lint, snapshots and files."""
    import litellm

    from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json

    stub = ("python -c \"import json; ids=json.load(open('/tmp/v68/scope.json'))['ids']; "
            "open('/tmp/v68/decisions.jsonl','w').write(''.join(json.dumps({'id': i, 'decision': 'kept', "
            "'note': 'preflight stub'})+chr(10) for i in ids))\" && python /tmp/v68/lint.py")
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        turn = sum(1 for m in kwargs.get("messages", []) if m.get("role") == "assistant")
        command = stub if turn == 0 else rp.SUBMIT
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Preflight.", "reasoning_content": "",
                "tool_calls": [{"id": f"pre-{len(calls)}", "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
            }}], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    scratch = root / "preflight-run"
    scratch.mkdir(exist_ok=True)
    for name in ("scripts", "configs"):
        if not (scratch / name).exists():
            (scratch / name).symlink_to(root / name, target_is_directory=True)
    first = next(r for r in selection["pairs"] if r["issue_count"] > 0)
    results = run_all(scratch, {**selection, "pairs": [first]}, config, "repair", 1, 1, model_patch=completion,
                      grade_on=False)
    result = results[pair_key(first)]
    record = read(scratch / "pairs" / pair_key(first) / "repair/episode/episode.json")
    passed = bool(result.get("status") == "not_graded" and result.get("lint_ok") and not result.get("code_changed")
                  and result["decisions"]["kept"] == first["issue_count"] and record.get("exit_status") == "Submitted")
    save(root / "preflight/readiness.json", {
        "passed": passed, "model_api_calls": 0, "simulated_responses": len(calls),
        "selection_sha256": sha256_json(selection), "stub_result": result, "finished_at": base.load_native(root).now()})
    if not passed:
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "control", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--map-root", type=Path)
    parser.add_argument("--resolve-root", type=Path)
    parser.add_argument("--note")
    parser.add_argument("--containers", type=int, default=6)
    parser.add_argument("--graders", type=int, default=3)
    parser.add_argument("--only", action="append")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        selection = prepare(root, args.review_root.resolve(), args.map_root.resolve(), args.resolve_root.resolve(),
                            args.note)
    else:
        selection, config = validate(root)
        if args.mode == "preflight":
            preflight(root, selection, config)
        elif args.mode == "control":
            run_all(root, selection, config, "control", args.containers, args.graders, only=args.only)
        else:
            from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json

            readiness = read(root / "preflight/readiness.json")
            if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
                raise ValueError("Run the zero-API preflight for this frozen selection first")
            if not os.environ.get("DEEPSEEK_API_KEY"):
                raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
            run_all(root, selection, config, "repair", args.containers, args.graders, only=args.only)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID, "pair_count": selection["pair_count"]}))


if __name__ == "__main__":
    main()
