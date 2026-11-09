"""Resume frozen failed cases at I/R, then score each result in the same worker.

Run from a NEW experiment root containing project/, selection.json and fork plans.
Credentials are inherited from the host environment and never written to disk.
"""
from __future__ import annotations

import collections
import contextvars
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

ROOT = Path(sys.argv[1]).resolve()
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

import docker
import requests
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec

from external_agents_eval.agents.mini_v62 import MiniBashEnvironment
from external_agents_eval.agents.mini_v63 import MiniFileReviewEnvironment

spec = importlib.util.spec_from_file_location("verified_generation", PROJECT / "scripts/swe_verified_v62/run.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)
SELECTION = json.loads((ROOT / "selection.json").read_text())
TASKS = {row["instance_id"]: row for row in map(json.loads, (ROOT / "manifests/public_tasks.jsonl").read_text().splitlines())}
REFS = {row["instance_id"]: row for row in json.loads((ROOT / "grader/reference_tasks.json").read_text())}
TASK_CONTEXT = contextvars.ContextVar("review_rerun_task", default="")
LOCK = threading.Lock()
IMAGE_LOCK = threading.Lock()
STATES = {row["instance_id"]: {"phase": "queued", "start_at": "I/R"} for row in SELECTION["cases"]}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(text):
    return str(text).replace(os.environ.get("DEEPSEEK_API_KEY", "__NO_KEY__"), "[REDACTED]")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(threading.get_ident()) + ".tmp")
    temporary.write_text(safe(json.dumps(value, ensure_ascii=False, indent=2)) + "\n")
    temporary.replace(path)


def update(tid, **changes):
    with LOCK:
        STATES[tid].update(changes, updated_at=now())
        save(ROOT / "cases" / tid / "status.json", STATES[tid])
        save(ROOT / "monitor-status.json", {
            "updated_at": now(), "task_count": len(STATES), "pipeline_workers": 3,
            "workflow": "v6.3", "scheduling": "review_then_official_evaluate_per_slot",
            "counts": dict(collections.Counter(row["phase"] for row in STATES.values())), "cases": STATES,
        })


def image(tid):
    return generation.image_name(tid)


def ensure_image(tid):
    # Bound concurrent pull disk pressure; review/evaluation still have three slots.
    with IMAGE_LOCK:
        client = docker.from_env(timeout=900)
        try:
            try:
                client.images.get(image(tid))
                return False
            except docker.errors.ImageNotFound:
                pass
            if shutil.disk_usage(ROOT).free < 15 * 1024**3:
                raise RuntimeError("Less than 15 GiB free before pulling task image")
            for attempt in range(1, 4):
                try:
                    update(tid, phase="image_preparation", image_attempt=attempt)
                    result = client.images.pull(image(tid))
                    save(ROOT / "cases" / tid / "image.json", {"name": image(tid), "id": result.id})
                    return True
                except docker.errors.APIError:
                    if attempt == 3:
                        raise
                    time.sleep(attempt * 3)
        finally:
            client.close()


# Preserve the already-used compatibility variant for Sphinx-9602 only. A/Solver
# are reused; its bounded tool observations do not change FT evidence or P0.
def bounded(execute):
    def call(self, action):
        result = execute(self, action)
        if TASK_CONTEXT.get() == "sphinx-doc__sphinx-9602" and len(result["output"]) > 65536:
            text = result["output"]
            digest = sha(text.encode())
            path = ROOT / "oversize-observations" / (digest + ".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            save(path.with_suffix(".json"), {"sha256": digest, "characters": len(text), "limit_chars": 65536})
            result["output"] = (text[:32768] + "\n[Output truncated; use a narrower command.]\n" + text[-32000:])
        return result
    return call


MiniBashEnvironment.execute = bounded(MiniBashEnvironment.execute)
MiniFileReviewEnvironment.execute = bounded(MiniFileReviewEnvironment.execute)

# Same host-only public environment-download fallback used in the recovered run.
stock_get = requests.get


def environment_get(url, *args, **kwargs):
    if not str(url).startswith("https://raw.githubusercontent.com/"):
        return stock_get(url, *args, **kwargs)
    kwargs.setdefault("timeout", 20)
    try:
        response = stock_get(url, *args, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException:
        raw = subprocess.run([
            "curl", "--silent", "--show-error", "--fail", "--location", "--retry", "2",
            "--connect-timeout", "10", "--max-time", "90", "--socks5-hostname", "127.0.0.1:19733", str(url),
        ], capture_output=True, timeout=200, check=False)
        if raw.returncode:
            raise RuntimeError("Public environment download failed: " + raw.stderr.decode(errors="replace")[-500:])
        response = requests.Response()
        response.status_code, response._content, response.url = 200, raw.stdout, url
        return response


requests.get = environment_get


def generate(item):
    tid = item["instance_id"]
    for attempt in range(1, 3):
        where = ROOT / "generation" / tid / f"attempt-{attempt:02d}"
        where.mkdir(parents=True, exist_ok=True)
        plan = where / "fork-plan.json"
        save(plan, {"task_id": tid, "start_at": "I/R", "source_run": item["source_root"]})
        update(tid, phase="reviewing", generation_attempt=attempt, generation_root=str(where))
        result = generation.run_one(TASKS[tid], SimpleNamespace(
            root=where, project=PROJECT, fork_plan=plan, workflow="v6.3",
        ))
        save(where / "run-result.json", result)
        if result.get("status") in {"generated", "already_generated"}:
            output = where / "outputs" / tid
            prediction = json.loads((output / "prediction.json").read_text())
            save(ROOT / "outputs" / tid / "result.json", {
                "status": "generated", "generation_root": str(where), "start_at": "I/R",
                "ledger_reused": True, "p0_reused": True,
            })
            return prediction
        error = result.get("error", "Unknown generation failure")
        transient = any(token in error for token in ["APIError", "Connection", "Timeout", "timed out", "EOF"])
        if not transient or attempt == 2:
            raise RuntimeError("Generation failed: " + error)
        update(tid, phase="retrying_generation", last_error=safe(error))


def chunks(patch):
    return ["diff --git " + chunk for chunk in patch.split("diff --git ")[1:]]


def destination(chunk):
    return shlex.split(chunk.splitlines()[0])[3].removeprefix("b/")


def grade(tid, prediction):
    original = prediction["model_patch"]
    scored = copy.deepcopy(prediction)
    additions = {destination(c) for c in chunks(REFS[tid].get("test_patch", ""))
                 if "\n--- /dev/null\n" in c or "\nnew file mode " in c}
    removed, keep = [], []
    for chunk in chunks(original):
        path = destination(chunk)
        if path in additions and ("\n--- /dev/null\n" in chunk or "\nnew file mode " in chunk):
            removed.append(path)
        else:
            keep.append(chunk)
    assert all(p.startswith("tests/") or "/tests/" in p for p in removed)
    if removed:
        scored["model_patch"] = "".join(keep)
    out = ROOT / "outputs" / tid
    save(out / "prediction.original.json", prediction)
    save(out / "prediction.json", scored)
    normalization = {"original_patch_sha256": sha(original.encode()),
                     "scored_patch_sha256": sha(scored["model_patch"].encode()),
                     "removed_official_test_collisions": removed, "production_code_unchanged": True}
    save(out / "scoring-normalization.json", normalization)
    if not scored["model_patch"].strip():
        update(tid, phase="empty_patch")
        return
    for attempt in range(1, 4):
        update(tid, phase="official_evaluation", scoring_attempt=attempt)
        run_id = ROOT.name + f"-g{attempt}"
        report = ROOT / "verifier/logs/run_evaluation" / run_id / scored["model_name_or_path"] / tid / "report.json"
        client = docker.from_env(timeout=900)
        try:
            if not report.exists():
                run_instance(make_test_spec(REFS[tid], namespace="swebench"), scored,
                             False, False, client, run_id, timeout=1800)
            if not report.exists():
                raise RuntimeError("Official scorer returned without report.json")
            row = json.loads(report.read_text())[tid]
            text = report.with_name("test_output.txt").read_text(errors="replace")
            anomalies = ["already exists in working directory", "patch does not apply",
                         "collected 0 items / 1 error", "Interrupted: 1 error during collection"]
            if any(token in text for token in anomalies):
                raise RuntimeError("Official test injection/collection anomaly")
            status = "resolved" if row["resolved"] else "unresolved"
            save(out / "evaluation.json", {"instance_id": tid, "status": status,
                "official_report_path": str(report), "finished_at": now(), "normalization": normalization,
                "test_counts": {k: {f: len(v[f]) for f in ["success", "failure"]}
                                for k, v in row["tests_status"].items()}})
            update(tid, phase=status, official_report_path=str(report), finished_at=now())
            return
        except Exception as error:
            save(out / f"scoring-error-{attempt}.json", {"error": safe(error), "at": now()})
            if attempt == 3:
                raise
        finally:
            client.close()


def run_case(item):
    tid = item["instance_id"]
    token = TASK_CONTEXT.set(tid)
    added = False
    try:
        added = ensure_image(tid)
        prediction = generate(item)
        grade(tid, prediction)
    except Exception as error:
        phase = "evaluation_error" if STATES[tid]["phase"] == "official_evaluation" else "generation_error"
        update(tid, phase=phase, error=safe(error), finished_at=now())
    finally:
        if added:
            client = docker.from_env()
            try:
                client.images.remove(image(tid), noprune=False)
            except docker.errors.APIError:
                pass
            finally:
                client.close()
        TASK_CONTEXT.reset(token)


def main():
    assert os.environ.get("DEEPSEEK_API_KEY"), "Missing host-only model credential"
    preserved = json.loads((ROOT / "preserved-original-sha256.json").read_text())
    assert all(sha(Path(p).read_bytes()) == h for p, h in preserved.items()), "Original stage artifacts changed"
    (ROOT / "verifier").mkdir(exist_ok=True)
    os.chdir(ROOT / "verifier")
    save(ROOT / "supervisor-status.json", {"phase": "running", "pid": os.getpid(),
        "started_at": now(), "pipeline_workers": 3, "task_count": len(STATES), "workflow": "v6.3"})
    for tid in STATES:
        update(tid, phase="queued")
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run_case, item) for item in SELECTION["cases"]]):
            future.result()
    unchanged = all(sha(Path(p).read_bytes()) == h for p, h in preserved.items())
    save(ROOT / "supervisor-status.json", {"phase": "finished", "pid": os.getpid(),
        "finished_at": now(), "counts": dict(collections.Counter(row["phase"] for row in STATES.values())),
        "original_artifacts_unchanged": unchanged})


if __name__ == "__main__":
    main()
