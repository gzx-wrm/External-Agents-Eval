"""V6.8 patch-conditioned review experiment (no Solver, no repair; user, 2026-11-08).

For each (task, failing candidate) pair: one review episode per V6.8.1 generation batch, each a
native Mini agent (M0's model, environment and templates; only the task section and limits
differ) in a fresh offline baseline container prepared exactly like M0's, with the candidate's
patch applied. The agent reviews the batch's V6.8.1 questions against the candidate and writes
review.jsonl (checked by v68_review_lint). prepare and preflight make no model calls; preflight
also checks that every candidate patch applies; run is explicit; outputs go to a new root; a
finished episode is never resampled.

usage: review.py prepare --root NEW --pairs pairs.json --v681-root ROOT --note TEXT
       review.py preflight --root ROOT
       review.py run --root ROOT [--containers N]
pairs.json: [{"task_id": ..., "candidate": ..., "dir": candidate output dir with model.patch}, ...]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.workflows.solve_review_repair import v68_review as rv  # noqa: E402
from external_agents_eval.workflows.solve_review_repair import v68_review_lint as rlint  # noqa: E402

EXPERIMENT_ID = "v6.8-patch-review"
RUNNER_VERSION = "v6.8-review-1"
FROZEN_SOURCES = (
    "scripts/deepswe_v68_gen/review.py",
    "scripts/deepswe_v68_gen/run.py",
    "external_agents_eval/workflows/solve_review_repair/v68_review.py",
    "external_agents_eval/workflows/solve_review_repair/v68_review_lint.py",
)
LINT_PATH = PROJECT / "external_agents_eval/workflows/solve_review_repair/v68_review_lint.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("v68_review_base", HERE / "run.py")  # container helpers of the V6.8.1 generation runner
read, save, sha = base.read, base.save, base.sha


def code_hashes():
    return {name: sha(PROJECT / name) for name in FROZEN_SOURCES}


def _sections(text):
    return [part for part in re.split(r"(?m)^(?=diff --git )", text) if part.startswith("diff --git ")]


def candidate_patch(directory):
    """The graded candidate diff without the image's environment-baseline sections."""
    directory = Path(directory)
    source = directory / "submitted.patch" if (directory / "submitted.patch").exists() else directory / "model.patch"
    text = source.read_text(errors="replace")
    environment = directory / "environment-baseline.patch"
    known = set(_sections(environment.read_text(errors="replace"))) if environment.exists() else set()
    return "".join(s for s in _sections(text) if s not in known), str(source)


def pair_key(row):
    return f"{row['task_id']}__{row['candidate']}"


def prepare(root, pairs_file, v681_root, note):
    import yaml

    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
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
    (root / "configs/review-mini.yaml").write_text(yaml.safe_dump(
        base.episode_config(config, rv.header(), rv.STEP_LIMIT, rv.WALL_TIME_SEC), allow_unicode=True, sort_keys=False))
    v681 = read(v681_root / "selection.json")
    tasks = {row["task_id"]: row for row in v681["tasks"]}
    (root / "patches").mkdir()
    rows = []
    for pair in json.loads(Path(pairs_file).read_text()):
        text, source = candidate_patch(pair["dir"])
        if not text.strip():
            raise ValueError(f"{pair_key(pair)}: empty candidate patch")
        patch_path = root / "patches" / f"{pair_key(pair)}.diff"
        patch_path.write_text(text)
        data_path = v681_root / "tasks" / pair["task_id"] / "agent" / "data.json"
        rows.append({"task_id": pair["task_id"], "candidate": pair["candidate"], "candidate_dir": pair["dir"],
                     "candidate_patch_source": source, "patch": str(patch_path), "patch_sha256": sha(patch_path),
                     "data": str(data_path), "data_sha256": sha(data_path),
                     "batches": rv.batches(read(data_path)), "task_row": tasks[pair["task_id"]]})
    selection = {"experiment_id": EXPERIMENT_ID, "version": rv.VERSION, "runner_version": RUNNER_VERSION,
                 "note": note, "v681_root": str(v681_root), "pair_count": len(rows), "pairs": rows,
                 "code_sha256": code_hashes(),
                 "native_source_sha256": {n: sha(root / "scripts" / n) for n in base.NATIVE_COPIES},
                 "config_sha256": {"review-mini.yaml": sha(root / "configs/review-mini.yaml")},
                 "limits": {"review": [rv.STEP_LIMIT, rv.WALL_TIME_SEC]}}
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
    if sha(root / "configs/review-mini.yaml") != selection["config_sha256"]["review-mini.yaml"]:
        raise ValueError("Episode configuration changed")
    for row in selection["pairs"]:
        if sha(row["patch"]) != row["patch_sha256"] or sha(row["data"]) != row["data_sha256"]:
            raise ValueError(f"{pair_key(row)}: frozen input changed")
    return selection, yaml.safe_load((root / "configs/review-mini.yaml").read_text())


APPLY = f"cd /app && git apply --whitespace=nowarn {rv.WORK_DIR}/candidate.diff"


def review_episode(native, root, row, config, output, files, *, model_patch=None, agent_on=True):
    """One review episode in a fresh baseline container with the candidate applied."""
    if (output / "episode.json").exists():
        return read(output / "episode.json")
    if output.exists():
        raise RuntimeError(f"{output} has a partial episode; not resampled")
    record = {"task_id": row["task_id"], "candidate": row["candidate"], "started_at": native.now()}
    environment = None
    try:
        environment, container = base.baseline_environment(native, root, row["task_row"], config, output / "container")
        for name, data in files.items():
            native.upload(container, data, f"{rv.WORK_DIR}/{name}")
        applied = environment.execute({"command": APPLY}, timeout=300)
        record["patch_applied"] = applied["returncode"] == 0
        (output / "apply.txt").write_text(applied["output"])
        if not record["patch_applied"]:
            record["exit_status"] = "patch_failed"
        elif agent_on:
            agent = native.DefaultAgent(native.get_model(config=config["model"]), environment, **config["agent"],
                                        output_path=output / "traj.json")
            try:
                if model_patch is not None:
                    with patch("litellm.completion", model_patch):
                        info = agent.run(row["task_row"]["task"]["instruction"])
                else:
                    info = agent.run(row["task_row"]["task"]["instruction"])
                record["exit_status"] = info.get("exit_status")
            except Exception as error:  # noqa: BLE001 - keep what the agent wrote
                record.update(exit_status="agent_error", error_type=type(error).__name__, error=str(error)[:2000])
            record["steps"] = sum(1 for m in getattr(agent, "messages", [])
                                  if isinstance(m, dict) and m.get("role") == "assistant")
            lint = environment.execute({"command": f"cd {rv.WORK_DIR} && python lint.py"}, timeout=120)
            record["lint_ok"] = lint["returncode"] == 0
            (output / "lint.txt").write_text(lint["output"])
            chunks, _ = container.get_archive(rv.WORK_DIR)
            base.untar_json(b"".join(chunks), output / "files")
    finally:
        base.close(native, environment)
    record["finished_at"] = native.now()
    save(output / "episode.json", record)
    return record


def batch_files(row, data, ids):
    return {"task.md": row["task_row"]["task"]["instruction"].encode(), "lint.py": LINT_PATH.read_bytes(),
            "candidate.diff": Path(row["patch"]).read_bytes(), "scope.json": json.dumps({"ids": ids}).encode(),
            "questions.jsonl": rv.question_lines(data, ids)}


def run_pair(native, root, row, config, containers, *, model_patch=None, only_first=False):
    out = root / "pairs" / pair_key(row)
    if (out / "result.json").exists():
        return read(out / "result.json")
    data = read(row["data"])
    groups = row["batches"][:1] if only_first else row["batches"]

    def one(index, ids):
        with containers:
            record = review_episode(native, root, row, config, out / f"batch-{index:02d}", batch_files(row, data, ids),
                                    model_patch=model_patch)
        ids_, rows, errors = (rlint.load(str(out / f"batch-{index:02d}" / "files"))
                              if (out / f"batch-{index:02d}" / "files" / "scope.json").exists() else (ids, [], []))
        return {"batch": index, "ids": ids, "record": record, "parse_errors": errors,
                "rows": [{k: v for k, v in r.items() if k != "_line"} for r in rlint.final_rows(ids, rows)]}

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        results = [f.result() for f in [pool.submit(one, i, ids) for i, ids in enumerate(groups, 1)]]
    rows = [r for result in results for r in result["rows"]]
    save(out / "review.json", {"pair": pair_key(row), "rows": rows})
    (out / "defects.md").write_text(rv.defects_markdown(data, rows))
    result = {"pair": pair_key(row), "status": "done", "questions": sum(len(r["ids"]) for r in results),
              "reviewed": len(rows), "defects": sum(r.get("status") == "defect" for r in rows),
              "unsure": sum(r.get("status") == "unsure" for r in rows),
              "batches": [{"batch": r["batch"], "questions": len(r["ids"]), "reviewed": len(r["rows"]),
                           **{k: r["record"].get(k) for k in ("patch_applied", "exit_status", "lint_ok", "steps")}}
                          for r in results],
              "finished_at": native.now()}
    save(out / "result.json", result)
    return result


def run_all(root, selection, config, max_containers, *, model_patch=None, only=None, only_first=False):
    native = base.load_native(root)
    containers = threading.BoundedSemaphore(max_containers)
    rows = [r for r in selection["pairs"] if not only or pair_key(r) in only]
    lock, results = threading.Lock(), {}

    def one(row):
        try:
            base.ensure_image(native, row["task_row"])
            result = run_pair(native, root, row, config, containers, model_patch=model_patch, only_first=only_first)
        except Exception as error:  # noqa: BLE001 - keep per-pair failures explicit
            result = {"pair": pair_key(row), "status": "stage_error", "error_type": type(error).__name__,
                      "error": str(error)[:2000]}
            save(root / "errors" / f"{pair_key(row)}.json", result)
        with lock:
            results[pair_key(row)] = result
            save(root / "result.json", {"experiment_id": EXPERIMENT_ID, "results": list(results.values())})
        print(json.dumps({k: result.get(k) for k in ("pair", "status", "reviewed", "defects", "unsure")}), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, len(rows))) as pool:
        for future in as_completed([pool.submit(one, row) for row in rows]):
            future.result()
    return results


def apply_checks(root, selection, config, max_containers):
    """Zero-API: every candidate patch must apply to the M0-prepared baseline."""
    native = base.load_native(root)
    containers = threading.BoundedSemaphore(max_containers)

    def one(row):
        with containers:
            base.ensure_image(native, row["task_row"])
            files = {"candidate.diff": Path(row["patch"]).read_bytes()}
            return pair_key(row), review_episode(native, root, row, config, root / "preflight/apply" / pair_key(row),
                                                 files, agent_on=False)

    with ThreadPoolExecutor(max_workers=len(selection["pairs"])) as pool:
        return dict(f.result() for f in [pool.submit(one, r) for r in selection["pairs"]])


def preflight(root, selection, config, max_containers):
    import litellm

    from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json

    applies = apply_checks(root, selection, config, max_containers)
    stub = ("python -c \"import json; ids=json.load(open('/tmp/v68/scope.json'))['ids']; "
            "open('/tmp/v68/review.jsonl','w').write(''.join(json.dumps({'id': i, 'status': 'unsure', "
            "'candidate_behavior': 'preflight stub'})+chr(10) for i in ids))\" && python /tmp/v68/lint.py")
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        turn = sum(1 for m in kwargs.get("messages", []) if m.get("role") == "assistant")
        command = stub if turn == 0 else rv.SUBMIT
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
    first = selection["pairs"][0]
    results = run_all(scratch, {**selection, "pairs": [first]}, config, 1, model_patch=completion, only_first=True)
    result = results[pair_key(first)]
    stub_ok = (result.get("status") == "done" and all(b["lint_ok"] for b in result["batches"])
               and result["reviewed"] == result["questions"])
    applied = {key: record.get("patch_applied") for key, record in applies.items()}
    save(root / "preflight/readiness.json", {
        "passed": bool(stub_ok and all(applied.values())), "stub_passed": bool(stub_ok), "patch_applied": applied,
        "model_api_calls": 0, "simulated_responses": len(calls), "selection_sha256": sha256_json(selection),
        "stub_result": result, "finished_at": base.load_native(root).now()})
    if not (stub_ok and all(applied.values())):
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--v681-root", type=Path)
    parser.add_argument("--note")
    parser.add_argument("--containers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        selection = prepare(root, args.pairs, args.v681_root.resolve(), args.note)
    else:
        selection, config = validate(root)
        if args.mode == "preflight":
            preflight(root, selection, config, args.containers)
        else:
            from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json

            readiness = read(root / "preflight/readiness.json")
            if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
                raise ValueError("Run the zero-API preflight for this frozen selection first")
            if not os.environ.get("DEEPSEEK_API_KEY"):
                raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
            run_all(root, selection, config, args.containers)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID, "pair_count": selection["pair_count"]}))


if __name__ == "__main__":
    main()
