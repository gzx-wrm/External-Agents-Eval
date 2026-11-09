"""V7 pilot: task-text acceptance tests as a submission gate (test generation only).

Per task: a native Mini test author writes new test files and a manifest on the frozen
baseline (no candidate, no hidden tests). The host then runs every manifest command in a
fresh container for each subject: the baseline, the M0 candidate, the official gold patch
(diagnostic only, never used by the gate) and up to two other graded candidates. A
candidate is rejected when a hard test fails on it twice. The official M0 reward is the
label. No Solver, review or official grading runs here.

Subcommands: prepare, preflight (zero API), run, report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import shlex
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.benchmarks.deepswe.submission_filter import strip_added_tests  # noqa: E402
from external_agents_eval.workflows.acceptance_gate import v7_testgen as tg  # noqa: E402

EXPERIMENT_ID = "v7-testgen-pilot"
SEED = 20261106
M0_ROOTS = {"batch1": Path("/root/mini-swe-deepswe-ft30-20261028"),
            "batch2": Path("/root/mini-swe-deepswe-heldout30-20261102"),
            "batch3": Path("/root/mini-swe-deepswe-batch3-30-20261105-w3-r1")}
BATCH1_COMPOSITE = M0_ROOTS["batch1"] / "composite-current.json"
BATCH3_PROVENANCE = Path("/root/v664-deepswe-batch3-30-20261105-w3-r1/m0-reference/provenance.json")
# Other graded candidates whose patches come from the same native snapshot protocol.
SUPPLEMENTARY = {
    "ft_m1": [Path("/root/ft_m1-deepswe-m0passed22-20261028-w3-r1/outputs"),
              Path("/root/ft_m1-deepswe-m0failed8-20261028-w3-r1/outputs")],
    "v6.3_p0": [Path("/root/v63-deepswe-heldout29-20261102-w3-r1/solver")],
    "v6.3_p1": [Path("/root/v63-deepswe-heldout29-20261102-w3-r1/outputs")],
    "v6.6.4_p0": [Path("/root/v664-deepswe-batch3-30-20261105-w3-r1/e2e/solver")],
    "v6.6.4_p1": [Path("/root/v664-deepswe-batch3-30-20261105-w3-r1/e2e/outputs")],
}
MAX_SUPPLEMENTARY = 2
# Tasks inspected in detail during the V6.6.4 analysis never enter the holdout.
ANALYZED_EXTRA = {"kgateway-consistent-hash-policy"}
SPLITS = {"iter": (20, 15), "holdout": (5, 5)}

TG_STEP_LIMIT = 150
TG_WALL_SEC = 2700
TG_COMMAND_TIMEOUT_SEC = 300
CMD_TIMEOUT_SEC = 600
SETUP_TIMEOUT_SEC = 1200
LABEL = "--label=external_eval=v7-testgen"
FROZEN_SOURCES = ("scripts/deepswe_v7_testgen/run.py",
                  "external_agents_eval/workflows/acceptance_gate/v7_testgen.py",
                  "external_agents_eval/benchmarks/deepswe/submission_filter.py")
NATIVE_FILES = {"m0_run.py": "run.py", "native_compat.py": "native_compat.py",
                "sanitize_repository.py": "sanitize_repository.py"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        text = text.replace(key, "[REDACTED]")
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: sha256(PROJECT / name) for name in FROZEN_SOURCES}


def load_native(root):
    if importlib.metadata.version("mini-swe-agent") != "2.4.6":
        raise ValueError("The native M0 protocol requires mini-SWE-agent 2.4.6")
    path = root / "scripts/m0_run.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("v7_native_m0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Labels and sampling
# ---------------------------------------------------------------------------

def _graded(directory):
    directory = Path(directory)
    result, patch_file = directory / "result.json", directory / "model.patch"
    if not (result.exists() and patch_file.exists()):
        return None
    reward = read(result).get("reward")
    return reward if reward in (0, 1) else None


def label_rows():
    rows = []
    sources = {"batch1": {Path(t["result_path"]).parent.name: Path(t["result_path"]).parent
                          for t in read(BATCH1_COMPOSITE)["tasks"]},
               "batch2": {p.name: p for p in sorted((M0_ROOTS["batch2"] / "outputs").iterdir())},
               "batch3": {tid: Path(src) for tid, src in read(BATCH3_PROVENANCE)["output_sources"].items()}}
    for batch, mapping in sources.items():
        root = M0_ROOTS[batch]
        tasks = {t["task_id"]: t for t in read(root / "manifests/public-tasks.json")}
        for tid, out in sorted(mapping.items()):
            reward = _graded(out)
            if reward is None:
                continue
            image_file = out / "image.json" if (out / "image.json").exists() else root / "outputs" / tid / "image.json"
            image = read(image_file)
            supplementary = []
            m0_key = hashlib.sha256(strip_added_tests((out / "model.patch").read_text())[0].encode()).hexdigest()
            seen = {m0_key}
            for name, dirs in SUPPLEMENTARY.items():
                for base in dirs:
                    reward_s = _graded(base / tid)
                    if reward_s is None:
                        continue
                    key = hashlib.sha256(strip_added_tests((base / tid / "model.patch").read_text())[0].encode()).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    supplementary.append({"name": name, "source": str(base / tid), "reward": reward_s})
            # Prefer candidates whose label differs from M0, then the rest; keep at most two.
            supplementary.sort(key=lambda s: (s["reward"] == reward, s["name"]))
            rows.append({"task_id": tid, "batch": batch, "label": reward, "task": tasks[tid],
                         "batch_root": str(root), "m0_source": str(out),
                         "image": {"id": image["id"], "repo_digests": image["repo_digests"]},
                         "gold_patch": str(root / "grader/tasks" / tid / "solution/solution.patch"),
                         "supplementary": supplementary[:MAX_SUPPLEMENTARY]})
    return rows


def sample(rows, seed):
    rng = random.Random(seed)
    analyzed = {r["task_id"] for r in rows if r["batch"] == "batch3"} | ANALYZED_EXTRA
    by_label = {label: sorted((r for r in rows if r["label"] == label), key=lambda r: r["task_id"])
                for label in (1, 0)}
    holdout = []
    for label, count in zip((1, 0), SPLITS["holdout"]):
        pool = [r for r in by_label[label] if r["task_id"] not in analyzed]
        holdout += rng.sample(pool, count)
    chosen = {r["task_id"] for r in holdout}
    iteration = []
    for label, count in zip((1, 0), SPLITS["iter"]):
        pool = [r for r in by_label[label] if r["task_id"] not in chosen]
        iteration += rng.sample(pool, count)
    return ([{**r, "split": "iter"} for r in iteration] + [{**r, "split": "holdout"} for r in holdout])


def prepare(root, seed, note, from_selection=None, reuse_from=None):
    import yaml

    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    native_sources = {name: {sha256(M0_ROOTS[b] / "scripts" / src) for b in M0_ROOTS}
                      for name, src in NATIVE_FILES.items()}
    if any(len(v) != 1 for v in native_sources.values()):
        raise ValueError("Native M0 scripts differ across batches")
    configs = [yaml.safe_load((M0_ROOTS[b] / "configs/native-deepseek-v4.1-flash.yaml").read_text())
               for b in M0_ROOTS]
    if any(c != configs[0] for c in configs):
        raise ValueError("Native M0 configurations differ across batches")
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    for target, source in NATIVE_FILES.items():
        (root / "scripts" / target).write_bytes((M0_ROOTS["batch3"] / "scripts" / source).read_bytes())
    (root / "configs").mkdir()
    native_config = configs[0]
    (root / "configs/native.yaml").write_text(yaml.safe_dump(native_config, allow_unicode=True, sort_keys=False))
    author = tg.tg_config(native_config, step_limit=TG_STEP_LIMIT, wall_time_sec=TG_WALL_SEC,
                          command_timeout_sec=TG_COMMAND_TIMEOUT_SEC, labels=[LABEL])
    (root / "configs/tg-mini.yaml").write_text(yaml.safe_dump(author, allow_unicode=True, sort_keys=False))
    if from_selection:
        rows = read(from_selection)["tasks"]
    else:
        rows = sample(label_rows(), seed)
    selection = {"experiment_id": EXPERIMENT_ID, "version": tg.VERSION, "note": note, "seed": seed,
                 "created_at": now(), "task_count": len(rows), "tasks": rows,
                 "counts": {s: {"m0_pass": sum(r["split"] == s and r["label"] == 1 for r in rows),
                                "m0_fail": sum(r["split"] == s and r["label"] == 0 for r in rows)}
                            for s in SPLITS},
                 "code_sha256": code_hashes(),
                 "native_sha256": {n: next(iter(v)) for n, v in native_sources.items()},
                 "tg_config_sha256": sha256(root / "configs/tg-mini.yaml"),
                 "limits": {"tg_step_limit": TG_STEP_LIMIT, "tg_wall_sec": TG_WALL_SEC,
                            "tg_command_timeout_sec": TG_COMMAND_TIMEOUT_SEC,
                            "cmd_timeout_sec": CMD_TIMEOUT_SEC, "setup_timeout_sec": SETUP_TIMEOUT_SEC},
                 "holdout_excludes": "batch3 tasks and " + ", ".join(sorted(ANALYZED_EXTRA)),
                 "from_selection": str(from_selection) if from_selection else None,
                 "reused_outputs_from": str(reuse_from) if reuse_from else None}
    if reuse_from:
        # Only the downstream gate changed: the test-author prompt and config must be identical.
        if sha256(reuse_from / "configs/tg-mini.yaml") != selection["tg_config_sha256"]:
            raise ValueError("Cannot reuse test-author outputs generated with a different prompt/config")
        import shutil
        for name in ("tg", "exec", "preflight"):
            if (reuse_from / name).exists():
                shutil.copytree(reuse_from / name, root / name, symlinks=True)
        selection["reused_tasks"] = sorted(p.name for p in (root / "tg").iterdir()) if (root / "tg").exists() else []
    save(root / "selection.json", selection)
    return selection


def validate(root):
    import yaml

    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID or selection["version"] != tg.VERSION:
        raise ValueError("Wrong experiment or version")
    if code_hashes() != selection["code_sha256"]:
        raise ValueError("Frozen experiment code changed; prepare a new root for a new version")
    for name, digest in selection["native_sha256"].items():
        if sha256(root / "scripts" / name) != digest:
            raise ValueError("Native source changed: " + name)
    if sha256(root / "configs/tg-mini.yaml") != selection["tg_config_sha256"]:
        raise ValueError("Test-author configuration changed")
    return (selection, yaml.safe_load((root / "configs/native.yaml").read_text()),
            yaml.safe_load((root / "configs/tg-mini.yaml").read_text()))


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

def require_image(client, row):
    image = client.images.get(row["image"]["id"])
    if image.id != row["image"]["id"] or row["image"]["repo_digests"][0] not in image.attrs.get("RepoDigests", []):
        raise ValueError("Local task image differs from the frozen M0 image")
    return image


def open_environment(native, row, environment_config):
    options = copy.deepcopy(environment_config)
    options["image"] = row["image"]["id"]
    task = row["task"]
    options["run_args"] = list(options["run_args"]) + [
        f"--cpus={task['cpus']}", f"--memory={task['memory_mb']}m"] + ([LABEL] if LABEL not in options["run_args"] else [])
    return native.get_environment(options)


def close_environment(native, environment):
    if environment is not None and environment.container_id:
        client = native.docker.from_env(timeout=300)
        try:
            client.containers.get(environment.container_id).remove(force=True)
        except native.docker.errors.NotFound:
            pass
        finally:
            client.close()
        environment.container_id = None


def run_command(environment, command, timeout_sec):
    wrapped = f"timeout -k 10 {int(timeout_sec)} bash -c {shlex.quote(command)}"
    started = time.time()
    result = environment.execute({"command": wrapped}, timeout=timeout_sec + 120)
    output = result.get("output") or ""
    timed_out = result.get("returncode") in (124, 137) or bool(result.get("exception_info"))
    return {"returncode": result.get("returncode"), "timeout": timed_out,
            "seconds": round(time.time() - started, 1), "output_tail": output[-3000:]}


# ---------------------------------------------------------------------------
# Test author
# ---------------------------------------------------------------------------

def tg_job(native, root, row, author_config, *, out=None, stub=None):
    tid, task = row["task_id"], row["task"]
    out = out or root / "tg" / tid
    if (out / "tg-result.json").exists():
        return read(out / "tg-result.json")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{tid}: a partial test-author attempt exists; not resampled")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    environment = None
    client = native.docker.from_env(timeout=1800)
    try:
        require_image(client, row)
        environment = open_environment(native, row, author_config["environment"])
        container = client.containers.get(environment.container_id)
        audit = native.prepare_repository(root, environment, container, task, out)
        agent = native.DefaultAgent(native.get_model(config=author_config["model"]), environment,
                                    **author_config["agent"], output_path=out / "tg.traj.json")
        if stub is not None:
            with patch("litellm.completion", stub):
                info = agent.run(task["instruction"])
        else:
            info = agent.run(task["instruction"])
        raw = environment.execute({"command": f"cat {tg.MANIFEST_PATH}"}, timeout=60)
        manifest_text = raw["output"] if raw["returncode"] == 0 else None
        snapshot = native.checked(environment, native.snapshot_command(audit["synthetic_commit"]))
    finally:
        close_environment(native, environment)
        client.close()
    environment_patch = (out / "environment-baseline.patch").read_text()
    test_patch, files, violations = tg.tg_test_patch(snapshot, environment_patch)
    (out / "tg-snapshot.patch").write_text(snapshot)
    (out / "tg.patch").write_text(test_patch)
    (out / "manifest.raw.json").write_text(manifest_text or "")
    manifest = tg.validate_manifest(tg.parse_manifest(manifest_text), task["instruction"], files)
    save(out / "manifest.json", manifest)
    stats = {}
    if (out / "tg.traj.json").exists():
        stats = (read(out / "tg.traj.json").get("info") or {}).get("model_stats") or {}
    record = {"task_id": tid, "version": tg.VERSION, "exit_status": info.get("exit_status"),
              "api_calls": stats.get("api_calls"), "cost": stats.get("instance_cost"),
              "seconds": round(time.time() - started, 1), "files": files, "violations": violations,
              "tests": len(manifest["tests"]), "valid_tests": sum(t["valid"] for t in manifest["tests"]),
              "quoted_tests": sum(t["valid"] and (t["quote_ok"] or t["kind"] == "existing")
                                  for t in manifest["tests"]),
              "manifest_problems": manifest["problems"], "finished_at": now()}
    save(out / "tg-result.json", record)
    return record


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

def subjects(row):
    items = [{"name": "base", "kind": "base", "reward": None},
             {"name": "m0", "kind": "candidate", "source": row["m0_source"], "reward": row["label"]},
             {"name": "gold", "kind": "gold", "reward": 1}]
    for index, s in enumerate(row["supplementary"]):
        items.append({"name": f"s{index + 1}-{s['name']}", "kind": "candidate", "source": s["source"],
                      "reward": s["reward"]})
    return items


def subject_job(native, root, row, subject, native_config, *, tg_dir=None, out=None):
    tid = row["task_id"]
    tg_dir = tg_dir or root / "tg" / tid
    out = out or root / "exec" / tid / subject["name"]
    if (out / "result.json").exists():
        return read(out / "result.json")
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(tg_dir / "manifest.json")
    test_patch = (tg_dir / "tg.patch").read_text()
    tests = [t for t in manifest["tests"] if t["valid"]]
    record = {"task_id": tid, "subject": subject["name"], "kind": subject["kind"],
              "reward": subject["reward"], "started_at": now(), "tests": {}, "rerun": {}}
    environment = None
    client = native.docker.from_env(timeout=1800)
    try:
        require_image(client, row)
        environment = open_environment(native, row, native_config["environment"])
        container = client.containers.get(environment.container_id)
        audit = native.prepare_repository(root, environment, container, row["task"], out)
        synthetic = audit["synthetic_commit"]

        def apply(data, name, *, reset=False):
            # Candidate patches carry the image baseline themselves; an empty one keeps it.
            if not data:
                return None
            if reset:
                native.checked(environment, f"git reset --hard -q {shlex.quote(synthetic)} && git clean -fdq")
            native.upload(container, data.encode(), f"/tmp/v7/{name}")
            result = environment.execute({"command": f"git apply --binary /tmp/v7/{name}"}, timeout=300)
            return None if result["returncode"] == 0 else result["output"][-2000:]

        error = None
        if subject["kind"] == "gold":
            error = apply(Path(row["gold_patch"]).read_text(), "gold.patch")
        elif subject["kind"] == "candidate":
            source_patch, removed = strip_added_tests((Path(subject["source"]) / "model.patch").read_text())
            record["candidate_removed_tests"] = removed
            error = apply(source_patch, "candidate.patch", reset=True)
        if error:
            record.update(status="subject_apply_error", error=error)
            return record
        error = apply(test_patch, "tests.patch")
        if error:
            record.update(status="test_apply_error", error=error)
            return record
        if manifest.get("setup"):
            record["setup"] = run_command(environment, manifest["setup"], SETUP_TIMEOUT_SEC)
        for test in tests:
            record["tests"][test["id"]] = run_command(environment, test["command"], CMD_TIMEOUT_SEC)
        if subject["kind"] != "base":
            for test in tests:
                if tg.status(record["tests"][test["id"]]) != "pass":
                    record["rerun"][test["id"]] = run_command(environment, test["command"], CMD_TIMEOUT_SEC)
        record["status"] = "executed"
        return record
    except Exception as error:  # noqa: BLE001 - keep per-subject infrastructure outcomes explicit
        record.update(status="infrastructure_error", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc()[-3000:])
        return record
    finally:
        close_environment(native, environment)
        client.close()
        record["finished_at"] = now()
        save(out / "result.json", record)


# ---------------------------------------------------------------------------
# Blind audit (v7.1)
# ---------------------------------------------------------------------------

AUDIT_URL = "https://api.deepseek.com/v1/chat/completions"
AUDIT_MODEL = "deepseek-flash"
AUDIT_TIMEOUT_SEC = 1800


def post_chat(messages):
    from urllib.request import Request

    from external_agents_eval.workflows.solve_review_repair.v66_transport import open_with_retry

    body = {"model": AUDIT_MODEL, "messages": messages, "thinking": {"type": "enabled"},
            "reasoning_effort": "high", "max_tokens": 65536, "stream": False}
    request = Request(AUDIT_URL, data=json.dumps(body, ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]})
    transport = []
    # Transport retry only (no valid response); a delivered response is never resampled.
    with open_with_retry(request, timeout=AUDIT_TIMEOUT_SEC, record=transport.append) as response:
        return json.loads(response.read().decode()), transport


def audit_job(root, row):
    tid = row["task_id"]
    tg_dir = root / "tg" / tid
    path = tg_dir / "audit.json"
    if path.exists():
        return read(path)
    if not (tg_dir / "manifest.json").exists():
        return None
    manifest = read(tg_dir / "manifest.json")
    ids = [t["id"] for t in manifest["tests"] if t["valid"]]
    messages = tg.audit_messages(row["task"]["instruction"], manifest, (tg_dir / "tg.patch").read_text())
    started = time.time()
    response, transport = post_chat(messages)
    choice = response["choices"][0]
    content = choice["message"].get("content") or ""
    (tg_dir / "audit.reasoning.txt").write_text(choice["message"].get("reasoning_content") or "")
    (tg_dir / "audit.output.txt").write_text(content)
    decisions, problems = tg.parse_audit(content, ids)
    record = {"task_id": tid, "version": tg.VERSION, "decisions": decisions, "problems": problems,
              "kept": sum(d["decision"] == "keep" for d in decisions.values()), "tests": len(ids),
              "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
              "transport": transport, "input_chars": sum(len(m["content"]) for m in messages),
              "seconds": round(time.time() - started, 1), "finished_at": now()}
    save(path, record)
    return record


def run_audit(root, selection, *, split, task_ids, workers):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the process environment")
    rows = [r for r in selection["tasks"] if (split == "all" or r["split"] == split)
            and (not task_ids or r["task_id"] in task_ids)]

    def one(row):
        try:
            result = audit_job(root, row)
            return row["task_id"], (result or {}).get("kept"), (result or {}).get("tests")
        except Exception as error:  # noqa: BLE001
            save(root / "errors" / f"audit-{row['task_id']}.json",
                 {"task_id": row["task_id"], "error": f"{type(error).__name__}: {error}"})
            return row["task_id"], None, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in rows]):
            print(json.dumps(future.result()), flush=True)


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------

def run_batch(root, selection, native_config, author_config, *, split, task_ids, tg_workers, exec_workers):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the process environment")
    if not read(root / "preflight/readiness.json")["passed"]:
        raise ValueError("Run the zero-API preflight first")
    native = load_native(root)
    rows = [r for r in selection["tasks"] if (split == "all" or r["split"] == split)
            and (not task_ids or r["task_id"] in task_ids)]
    lock = threading.Lock()
    state = {r["task_id"]: {"tg": "queued", "subjects": {}} for r in rows}

    def update(tid, **values):
        with lock:
            state[tid].update(values)
            save(root / "monitor-status.json", {"updated_at": now(), "split": split, "tasks": state})

    def author(row):
        tid = row["task_id"]
        update(tid, tg="running")
        try:
            result = tg_job(native, root, row, author_config)
            update(tid, tg=result.get("exit_status") or "done", tests=result.get("valid_tests"))
            return result
        except Exception as error:  # noqa: BLE001
            failure = {"task_id": tid, "status": "tg_error", "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()[-3000:]}
            save(root / "errors" / f"tg-{tid}.json", failure)
            update(tid, tg="tg_error")
            return None

    def execute(row, subject):
        tid = row["task_id"]
        result = subject_job(native, root, row, subject, native_config)
        with lock:
            state[tid]["subjects"][subject["name"]] = result.get("status")
        update(tid)
        return result

    pending = []
    with ThreadPoolExecutor(max_workers=tg_workers) as authors, \
            ThreadPoolExecutor(max_workers=exec_workers) as runners:
        futures = {authors.submit(author, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            if future.result() is None:
                continue
            for subject in subjects(row):
                pending.append(runners.submit(execute, row, subject))
        for future in as_completed(pending):
            future.result()
    save(root / f"run-{split}-finished.json", {"finished_at": now(), "tasks": len(rows)})


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

VARIANTS = ("primary", "checked", "audited", "audited_checked", "audited_checked_two_or_more",
            "new_only", "two_or_more", "gold_filtered")


def task_report(root, row):
    tid = row["task_id"]
    tg_result = read(root / "tg" / tid / "tg-result.json") if (root / "tg" / tid / "tg-result.json").exists() else None
    if tg_result is None:
        return {"task_id": tid, "status": "no_tests"}
    manifest = read(root / "tg" / tid / "manifest.json")
    tests = manifest["tests"]
    results = {}
    for subject in subjects(row):
        path = root / "exec" / tid / subject["name"] / "result.json"
        results[subject["name"]] = read(path) if path.exists() else None
    base = results.get("base") or {}
    if base.get("status") != "executed":
        return {"task_id": tid, "status": "base_not_executed", "tg": tg_result}
    classes = {t["id"]: tg.classify(t, base["tests"].get(t["id"])) for t in tests}
    hard_ids = {tid_ for tid_, c in classes.items() if c == "hard"}
    gold = results.get("gold") or {}
    gold_fail = set()
    if gold.get("status") == "executed":
        gold_fail = {t for t in hard_ids
                     if tg.status(gold["tests"].get(t)) != "pass" and tg.status(gold["rerun"].get(t)) != "pass"}
    audit_path = root / "tg" / tid / "audit.json"
    decisions = read(audit_path)["decisions"] if audit_path.exists() else None
    dropped = ({t for t, d in decisions.items() if d["decision"] != "keep"} | {t["id"] for t in tests if t["id"] not in decisions}
               if decisions is not None else None)
    instruction = row["task"]["instruction"]
    guessed = {t["id"]: tg.guessed_names((base["tests"].get(t["id"]) or {}).get("output_tail", ""), instruction)
               for t in tests if classes[t["id"]] == "hard" and t["kind"] == "new"}
    guessed_ids = {t for t, names in guessed.items() if names}
    m0r = results.get("m0") or {}
    per_test = [{"id": t["id"], "kind": t["kind"], "category": t["category"], "class": classes[t["id"]],
                 "gold": (None if gold.get("status") != "executed" else t["id"] not in gold_fail),
                 "m0_pass": (None if m0r.get("status") != "executed" else
                             tg.status(m0r["tests"].get(t["id"])) == "pass" or tg.status(m0r["rerun"].get(t["id"])) == "pass"),
                 "guessed": guessed.get(t["id"]) or [],
                 "audit": (decisions or {}).get(t["id"], {}).get("decision") if decisions is not None else None,
                 "criteria": (decisions or {}).get(t["id"], {}).get("criteria") if decisions is not None else None}
                for t in tests if classes[t["id"]] == "hard"]
    verdicts = {}
    for subject in subjects(row):
        if subject["kind"] == "base":
            continue
        r = results.get(subject["name"])
        if not r or r.get("status") != "executed":
            verdicts[subject["name"]] = {"status": (r or {}).get("status", "missing"), "reward": subject["reward"]}
            continue
        primary = tg.candidate_verdict(tests, base["tests"], r["tests"], r["rerun"])
        new_only = tg.candidate_verdict([t for t in tests if t["kind"] == "new"], base["tests"], r["tests"], r["rerun"])
        filtered = tg.candidate_verdict(tests, base["tests"], r["tests"], r["rerun"], exclude=gold_fail)
        audited = (tg.candidate_verdict(tests, base["tests"], r["tests"], r["rerun"], exclude=dropped)
                   if dropped is not None else None)
        checked = tg.candidate_verdict(tests, base["tests"], r["tests"], r["rerun"], exclude=guessed_ids)
        both = (tg.candidate_verdict(tests, base["tests"], r["tests"], r["rerun"], exclude=dropped | guessed_ids)
                if dropped is not None else None)
        verdicts[subject["name"]] = {
            "status": "executed", "reward": subject["reward"], "kind": subject["kind"],
            "primary": primary["reject"], "new_only": new_only["reject"],
            "two_or_more": len(primary["failing_hard"]) >= 2, "gold_filtered": filtered["reject"],
            "audited": audited["reject"] if audited else None,
            "audited_two_or_more": len(audited["failing_hard"]) >= 2 if audited else None,
            "audited_failing": audited["failing_hard"] if audited else None,
            "checked": checked["reject"],
            "audited_checked": both["reject"] if both else None,
            "audited_checked_two_or_more": len(both["failing_hard"]) >= 2 if both else None,
            "audited_checked_failing": both["failing_hard"] if both else None,
            "failing_hard": primary["failing_hard"], "flaky": primary["flaky"],
            "failing_gold_passing": [t for t in primary["failing_hard"] if t not in gold_fail]}
    counts = {c: sum(v == c for v in classes.values()) for c in sorted(set(classes.values()))}
    return {"task_id": tid, "batch": row["batch"], "split": row["split"], "label": row["label"],
            "status": "ok", "tg": {k: tg_result.get(k) for k in ("exit_status", "api_calls", "cost", "seconds",
                                                               "tests", "valid_tests", "violations")},
            "per_test": per_test, "audit_kept": (None if dropped is None else len(hard_ids - dropped)),
            "guessed_tests": len(guessed_ids),
            "classes": counts, "hard": len(hard_ids), "gold_failing_hard": sorted(gold_fail),
            "gold_hard_pass_rate": (round(1 - len(gold_fail) / len(hard_ids), 3) if hard_ids and gold.get("status") == "executed" else None),
            "gold_status": gold.get("status"), "verdicts": verdicts}


def rates(rows, subject_filter):
    out = {}
    for variant in VARIANTS:
        chosen = [v for r in rows for name, v in r.get("verdicts", {}).items()
                  if subject_filter(name, v) and v.get("status") == "executed" and v.get(variant) is not None]
        fail = [v for v in chosen if v["reward"] == 0]
        ok = [v for v in chosen if v["reward"] == 1]
        out[variant] = {"detected": sum(v[variant] for v in fail), "failing_candidates": len(fail),
                        "false_rejects": sum(v[variant] for v in ok), "passing_candidates": len(ok),
                        "detection_rate": round(sum(v[variant] for v in fail) / len(fail), 3) if fail else None,
                        "false_reject_rate": round(sum(v[variant] for v in ok) / len(ok), 3) if ok else None}
    return out


def test_stats(rows):
    """Gold pass rate and audit decisions of hard tests, by kind and by category."""
    tests = [t for r in rows for t in r.get("per_test", []) if t["gold"] is not None]

    def summarize(group):
        kept = [t for t in group if t["audit"] == "keep"]
        dropped = [t for t in group if t["audit"] == "drop"]
        return {"hard": len(group),
                "gold_fail_rate": round(sum(not t["gold"] for t in group) / len(group), 3) if group else None,
                "audit_keep_rate": round(len(kept) / len(group), 3) if group and any(t["audit"] for t in group) else None,
                "gold_fail_rate_kept": round(sum(not t["gold"] for t in kept) / len(kept), 3) if kept else None,
                "gold_fail_rate_dropped": round(sum(not t["gold"] for t in dropped) / len(dropped), 3) if dropped else None}

    keys = {"kind": sorted({t["kind"] for t in tests}), "category": sorted({str(t["category"]) for t in tests})}
    return {"all": summarize(tests),
            "by_kind": {k: summarize([t for t in tests if t["kind"] == k]) for k in keys["kind"]},
            "by_category": {c: summarize([t for t in tests if str(t["category"]) == c]) for c in keys["category"]}}


def report(root, selection, split):
    rows = [task_report(root, r) for r in selection["tasks"] if split == "all" or r["split"] == split]
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {
        "version": selection["version"], "split": split, "tasks": len(rows), "reported": len(ok),
        "not_reported": [(r["task_id"], r["status"]) for r in rows if r["status"] != "ok"],
        "m0": rates(ok, lambda name, v: name == "m0"),
        "gold": {variant: {k: v[k] for k in ("false_rejects", "passing_candidates", "false_reject_rate")}
                 for variant, v in rates(ok, lambda name, v: name == "gold").items()},
        "test_stats": test_stats(ok),
        "supplementary": rates(ok, lambda name, v: name.startswith("s")),
        "all_candidates": rates(ok, lambda name, v: v.get("kind") == "candidate"),
        "tg_cost": round(sum((r["tg"].get("cost") or 0) for r in ok), 3),
        "tg_api_calls": sum((r["tg"].get("api_calls") or 0) for r in ok),
        "mean_hard_tests": round(sum(r["hard"] for r in ok) / len(ok), 1) if ok else None,
        "gold_hard_pass_rate_mean": (round(sum(r["gold_hard_pass_rate"] for r in ok if r["gold_hard_pass_rate"] is not None)
                                           / max(1, sum(r["gold_hard_pass_rate"] is not None for r in ok)), 3)),
    }
    save(root / f"report-{split}.json", {"summary": summary, "tasks": rows, "generated_at": now()})
    lines = [f"# {selection['version']} test-generation pilot ({split})", "",
             "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
             "| task | batch | M0 | hard | gold pass | M0 reject | failing hard (gold-passing) | gold reject | supplementary |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: (r["label"], r["task_id"])):
        m0 = r["verdicts"].get("m0", {})
        supp = "; ".join(f"{n}:{v.get('reward')}->{'R' if v.get('primary') else 'A'}" for n, v in r["verdicts"].items()
                         if n.startswith("s") and v.get("status") == "executed")
        lines.append(f"| {r['task_id']} | {r['batch']} | {r['label']} | {r['hard']} (kept {r.get('audit_kept')}) | "
                     f"{r['gold_hard_pass_rate']} | {'REJECT' if m0.get('primary') else 'accept'}"
                     f"/{'REJECT' if m0.get('audited') else 'accept' if m0.get('audited') is not None else '-'} | "
                     f"{len(m0.get('failing_hard', []))} ({len(m0.get('failing_gold_passing', []))}) | "
                     f"{r['verdicts'].get('gold', {}).get('primary')}/{r['verdicts'].get('gold', {}).get('audited')} | {supp} |")
    (root / f"report-{split}.md").write_text("\n".join(lines) + "\n")
    return summary


# ---------------------------------------------------------------------------
# Zero-API preflight
# ---------------------------------------------------------------------------

def preflight(root, selection, native_config, author_config):
    import litellm

    native = load_native(root)
    row = next(r for r in selection["tasks"] if r["split"] == "iter")
    quote = tg.squash(row["task"]["instruction"])[:60]
    manifest = {"setup": None, "requirements": [{"id": "R1", "quote": quote}], "tests": [
        {"id": "T1", "requirements": ["R1"], "kind": "preserve", "category": "basic",
         "files": ["tests/v7_preflight_check.sh"], "command": "bash tests/v7_preflight_check.sh",
         "asserts": "preflight file runs"},
        {"id": "T2", "requirements": ["R1"], "kind": "new", "category": "basic",
         "files": ["tests/v7_preflight_check.sh"],
         "command": "bash tests/v7_preflight_check.sh && test -f /app/.v7-preflight-never",
         "asserts": "always fails"},
        {"id": "T3", "requirements": [], "kind": "existing", "category": "preservation", "files": [],
         "command": "git -C /app rev-parse HEAD >/dev/null", "asserts": "existing command"}]}
    commands = ["mkdir -p /app/tests /tmp/v7 && printf 'exit 0\\n' > /app/tests/v7_preflight_check.sh && "
                f"cat > {tg.MANIFEST_PATH} <<'EOF'\n{json.dumps(manifest)}\nEOF",
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        command = commands[min(len(calls), len(commands)) - 1]
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Preflight.", "reasoning_content": "",
                "tool_calls": [{"id": f"pre-{len(calls)}", "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]}}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    out = root / "preflight"
    tg_result = tg_job(native, root, row, author_config, out=out / "tg", stub=completion)
    results = {s["name"]: subject_job(native, root, row, s, native_config, tg_dir=out / "tg", out=out / "exec" / s["name"])
               for s in subjects(row)[:3]}
    manifest_checked = read(out / "tg/manifest.json")
    base = results["base"]
    verdicts = {name: tg.candidate_verdict(manifest_checked["tests"], base["tests"], r["tests"], r["rerun"])
                for name, r in results.items() if name != "base" and r.get("status") == "executed"}
    classes = {t["id"]: tg.classify(t, base["tests"].get(t["id"])) for t in manifest_checked["tests"]}
    passed = (len(calls) == 2 and tg_result["files"] == ["tests/v7_preflight_check.sh"]
              and not tg_result["violations"] and all(r.get("status") == "executed" for r in results.values())
              and classes == {"T1": "hard", "T2": "hard", "T3": "hard"}
              and all(v["failing_hard"] == ["T2"] for v in verdicts.values()))
    save(out / "readiness.json", {"passed": passed, "model_api_calls": 0, "simulated_responses": len(calls),
                                  "task_id": row["task_id"], "tg": tg_result, "classes": classes,
                                  "subjects": {n: r.get("status") for n, r in results.items()},
                                  "verdicts": verdicts, "finished_at": now()})
    if not passed:
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run", "audit", "report", "labels"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--note")
    parser.add_argument("--from-selection", type=Path)
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--split", default="iter", choices=["iter", "holdout", "all"])
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--tg-workers", type=int, default=6)
    parser.add_argument("--exec-workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "labels":
        rows = label_rows()
        print(json.dumps({"rows": len(rows), "pass": sum(r["label"] == 1 for r in rows),
                          "fail": sum(r["label"] == 0 for r in rows),
                          "with_supplementary": sum(bool(r["supplementary"]) for r in rows)}))
        return
    root = args.root.resolve()
    if args.mode == "prepare":
        selection = prepare(root, args.seed, args.note, args.from_selection,
                            args.reuse_from.resolve() if args.reuse_from else None)
        print(json.dumps({"mode": "prepare", "version": selection["version"], "counts": selection["counts"]}))
        return
    selection, native_config, author_config = validate(root)
    if args.mode == "preflight":
        preflight(root, selection, native_config, author_config)
    elif args.mode == "audit":
        run_audit(root, selection, split=args.split, task_ids=args.task_id, workers=args.audit_workers)
    elif args.mode == "run":
        run_batch(root, selection, native_config, author_config, split=args.split, task_ids=args.task_id,
                  tg_workers=args.tg_workers, exec_workers=args.exec_workers)
    else:
        print(json.dumps(report(root, selection, args.split), ensure_ascii=False, indent=2))
    print(json.dumps({"mode": args.mode, "version": selection["version"]}))


if __name__ == "__main__":
    main()
