"""V6.8 reading labels: observe which reading an implementation follows (no Solver; user, 2026-11-08).

Used with the reference solutions to build evaluation-only labels for the V6.8.1 questions. Per
(task, implementation) pair: one observation episode per V6.8.1 generation batch, each a native
Mini agent (M0's model, environment and templates; only the task section and limits differ) in a
fresh offline baseline container with the implementation's patch applied; it writes labels.jsonl
(checked by v68_observe_lint). prepare and preflight make no model calls; preflight checks that
every patch applies; run is explicit; outputs go to a new root; a finished episode is never
resampled. Container, patch and episode helpers come from review.py.

usage: observe.py prepare --root NEW --pairs pairs.json --v681-root ROOT --note TEXT
       observe.py preflight --root ROOT [--containers N]
       observe.py run --root ROOT [--containers N]
pairs.json: [{"task_id": ..., "candidate": "gold", "dir": directory holding model.patch}, ...]
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.workflows.solve_review_repair import v68_observe as ob  # noqa: E402
from external_agents_eval.workflows.solve_review_repair import v68_observe_lint as olint  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rev = _load("v68_observe_review", HERE / "review.py")
base, read, save, sha, pair_key = rev.base, rev.read, rev.save, rev.sha, rev.pair_key

EXPERIMENT_ID = "v6.8-reading-labels"
RUNNER_VERSION = "v6.8-observe-1"
CONFIG = "observe-mini.yaml"
FROZEN_SOURCES = (
    "scripts/deepswe_v68_gen/observe.py",
    "scripts/deepswe_v68_gen/review.py",
    "scripts/deepswe_v68_gen/run.py",
    "external_agents_eval/workflows/solve_review_repair/v68_observe.py",
    "external_agents_eval/workflows/solve_review_repair/v68_observe_lint.py",
    "external_agents_eval/workflows/solve_review_repair/v68_review.py",
)
LINT_PATH = PROJECT / "external_agents_eval/workflows/solve_review_repair/v68_observe_lint.py"


def code_hashes():
    return {name: sha(PROJECT / name) for name in FROZEN_SOURCES}


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
    (root / "configs" / CONFIG).write_text(yaml.safe_dump(
        base.episode_config(config, ob.header(), ob.STEP_LIMIT, ob.WALL_TIME_SEC), allow_unicode=True, sort_keys=False))
    tasks = {row["task_id"]: row for row in read(v681_root / "selection.json")["tasks"]}
    (root / "patches").mkdir()
    rows = []
    for pair in json.loads(Path(pairs_file).read_text()):
        text, source = rev.candidate_patch(pair["dir"])
        if not text.strip():
            raise ValueError(f"{pair_key(pair)}: empty patch")
        patch_path = root / "patches" / f"{pair_key(pair)}.diff"
        patch_path.write_text(text)
        data_path = v681_root / "tasks" / pair["task_id"] / "agent" / "data.json"
        rows.append({"task_id": pair["task_id"], "candidate": pair["candidate"], "candidate_dir": pair["dir"],
                     "candidate_patch_source": source, "patch": str(patch_path), "patch_sha256": sha(patch_path),
                     "data": str(data_path), "data_sha256": sha(data_path),
                     "batches": rev.rv.batches(read(data_path)), "task_row": tasks[pair["task_id"]]})
    selection = {"experiment_id": EXPERIMENT_ID, "version": ob.VERSION, "runner_version": RUNNER_VERSION,
                 "note": note, "v681_root": str(v681_root), "pair_count": len(rows), "pairs": rows,
                 "code_sha256": code_hashes(),
                 "native_source_sha256": {n: sha(root / "scripts" / n) for n in base.NATIVE_COPIES},
                 "config_sha256": {CONFIG: sha(root / "configs" / CONFIG)},
                 "limits": {"observe": [ob.STEP_LIMIT, ob.WALL_TIME_SEC]}}
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
        if sha(row["patch"]) != row["patch_sha256"] or sha(row["data"]) != row["data_sha256"]:
            raise ValueError(f"{pair_key(row)}: frozen input changed")
    return selection, yaml.safe_load((root / "configs" / CONFIG).read_text())


def batch_files(row, data, ids):
    return {"task.md": row["task_row"]["task"]["instruction"].encode(), "lint.py": LINT_PATH.read_bytes(),
            "candidate.diff": Path(row["patch"]).read_bytes(), "scope.json": json.dumps({"ids": ids}).encode(),
            "questions.jsonl": ob.question_lines(data, ids)}


def run_pair(native, root, row, config, containers, *, model_patch=None, only_first=False):
    out = root / "pairs" / pair_key(row)
    if (out / "result.json").exists():
        return read(out / "result.json")
    data = read(row["data"])
    groups = row["batches"][:1] if only_first else row["batches"]

    def one(index, ids):
        target = out / f"batch-{index:02d}"
        with containers:
            record = rev.review_episode(native, root, row, config, target, batch_files(row, data, ids),
                                        model_patch=model_patch)
        rows = []
        if (target / "files" / "scope.json").exists():
            scope_ids, _, raw, _ = olint.load(str(target / "files"))
            rows = [{k: v for k, v in r.items() if k != "_line"} for r in olint.final_rows(scope_ids, raw)]
        return {"batch": index, "ids": ids, "record": record, "rows": rows}

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        results = [f.result() for f in [pool.submit(one, i, ids) for i, ids in enumerate(groups, 1)]]
    rows = [r for result in results for r in result["rows"]]
    save(out / "labels.json", {"pair": pair_key(row), "rows": rows})
    result = {"pair": pair_key(row), "status": "done", "questions": sum(len(r["ids"]) for r in results),
              "labelled": len(rows), "observed": sum(r.get("status") == "observed" for r in rows),
              "not_applicable": sum(r.get("status") == "not_applicable" for r in rows),
              "undetermined": sum(r.get("status") == "undetermined" for r in rows),
              "batches": [{"batch": r["batch"], "questions": len(r["ids"]), "labelled": len(r["rows"]),
                           **{k: r["record"].get(k) for k in ("patch_applied", "exit_status", "lint_ok", "steps")}}
                          for r in results],
              "finished_at": native.now()}
    save(out / "result.json", result)
    return result


def run_all(root, selection, config, max_containers, *, model_patch=None, only_first=False):
    native = base.load_native(root)
    containers = threading.BoundedSemaphore(max_containers)
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
        print(json.dumps({k: result.get(k) for k in ("pair", "status", "labelled", "observed", "undetermined")}),
              flush=True)

    with ThreadPoolExecutor(max_workers=max(1, len(selection["pairs"]))) as pool:
        for future in as_completed([pool.submit(one, row) for row in selection["pairs"]]):
            future.result()
    return results


def preflight(root, selection, config, max_containers):
    import litellm

    from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json

    applies = rev.apply_checks(root, selection, config, max_containers)
    stub = ("python -c \"import json; ids=json.load(open('/tmp/v68/scope.json'))['ids']; "
            "open('/tmp/v68/labels.jsonl','w').write(''.join(json.dumps({'id': i, 'status': 'undetermined'})"
            "+chr(10) for i in ids))\" && python /tmp/v68/lint.py")
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        turn = sum(1 for m in kwargs.get("messages", []) if m.get("role") == "assistant")
        command = stub if turn == 0 else ob.SUBMIT
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
    result = run_all(scratch, {**selection, "pairs": [first]}, config, 1, model_patch=completion,
                     only_first=True)[pair_key(first)]
    stub_ok = (result.get("status") == "done" and all(b["lint_ok"] for b in result["batches"])
               and result["labelled"] == result["questions"])
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
