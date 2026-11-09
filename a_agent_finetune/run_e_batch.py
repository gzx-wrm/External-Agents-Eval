"""Prepare and run E0 tasks with frozen code and one GPU request at a time."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from a_agent_finetune.run_e_inference import FrozenWorkspace
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_tree(repository: Path, revision: str, destination: Path) -> dict:
    """Export actual Git blobs, including files marked export-ignore and without export-subst."""
    raw = subprocess.check_output(["git", "-C", str(repository), "ls-tree", "-rz", revision])
    entries, gitlinks = [], []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if kind == b"blob":
            entries.append((mode, oid, os.fsdecode(name)))
        else:
            gitlinks.append(os.fsdecode(name))
            (destination / os.fsdecode(name)).mkdir(parents=True, exist_ok=True)
    blobs = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "--batch"],
        input=b"\n".join(e[1] for e in entries) + b"\n",
        capture_output=True,
        check=True,
    ).stdout
    offset = 0
    for mode, oid, name in entries:
        end = blobs.index(b"\n", offset)
        actual_oid, kind, size = blobs[offset:end].split()
        if actual_oid != oid or kind != b"blob":
            raise ValueError(f"Cannot export {revision}:{name}")
        offset = end + 1
        data = blobs[offset : offset + int(size)]
        offset += int(size) + 1
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == b"120000":
            path.symlink_to(os.fsdecode(data))
        else:
            path.write_bytes(data)
            path.chmod(0o755 if mode == b"100755" else 0o644)
    return {"file_count": len(entries), "gitlinks_without_materialized_content": gitlinks}


def prepare(args):
    root = args.output.resolve()
    source = read_json(args.records / "manifest.json")
    if source["sample_count"] != 30 or len(source["tasks"]) != 30:
        raise ValueError("Expected the original 30-task development set")
    smoke = None
    if args.reuse_smoke:
        smoke = read_json(args.reuse_smoke / "manifest.json")
        if smoke["final_state"] != "A_COMPLETED":
            raise ValueError("The reused A result is not complete")
        for name in ("inference-config.json", "controller-config.json"):
            old = read_json(args.reuse_smoke / "inputs" / name)
            current = read_json(HERE / "e_smoke" / name.replace("-", "_"))
            if name == "controller-config.json":
                old.pop("role_timeout_sec", None)
            if old != current:
                raise ValueError(f"Smoke configuration differs: {name}")
        experiment = read_json(args.reuse_smoke / "inputs/experiment.json")
        if (
            experiment["timeout_sec"] != args.timeout_sec
            or experiment["max_tool_calls"] != args.max_tool_calls
            or experiment.get("loop_detection", True) != args.loop_detection
            or experiment.get("loop_repeat_limit", 8) != args.loop_repeat_limit
            or experiment.get("loop_max_cycle_length", 8) != args.loop_max_cycle_length
        ):
            raise ValueError("Smoke budgets differ from the batch budgets")
        if (args.reuse_smoke / "inputs/system-prompt.txt").read_bytes() != (
            HERE / "e_prompt_v1.txt"
        ).read_bytes():
            raise ValueError("Smoke prompt differs from the batch prompt")
    root.mkdir(parents=True, exist_ok=False)
    model_list = json.load(urlopen("http://127.0.0.1:3000/v1/models", timeout=15))
    model_root = next(m["root"] for m in model_list["data"] if m["id"] == "qwen35-9b")
    runtime = root / "runtime"
    shutil.copytree(
        PROJECT / "external_agents_eval",
        runtime / "external_agents_eval",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    package = runtime / "a_agent_finetune"
    package.mkdir()
    for name in ("run_e_inference.py", "run_e_batch.py", "e_prompt_v1.txt"):
        shutil.copy2(HERE / name, package / name)
    shutil.copytree(HERE / "e_smoke", package / "e_smoke")
    digest = hashlib.sha256()
    for path in sorted(runtime.rglob("*")):
        if path.is_file():
            digest.update(
                path.relative_to(runtime).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
            )
    state = {
        "schema_version": "e0-batch-1",
        "status": "PREPARING",
        "source_batch": source["source_batch"],
        "sample_count": 30,
        "new_task_count": len(source["tasks"]) - bool(smoke),
        "concurrency": 1,
        "timeout_sec": args.timeout_sec,
        "max_tool_calls": args.max_tool_calls,
        "loop_detection": args.loop_detection,
        "loop_repeat_limit": args.loop_repeat_limit,
        "loop_max_cycle_length": args.loop_max_cycle_length,
        "runtime_sha256": digest.hexdigest(),
        "model_root": model_root,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "current_task": None,
        "tasks": [],
    }
    write_json(root / "batch.json", state)
    for item in source["tasks"]:
        task_id, revision = item["task_id"], item["base_commit"]
        record = read_json(args.records / "records" / f"{task_id}.json")
        if record["base_commit"] != revision:
            raise ValueError(f"Record revision differs: {task_id}")
        task_input = {
            "task": {key: record["task"][key] for key in ("title", "description")},
            "managed_delivery": record.get("managed_delivery", []),
        }
        write_json(root / "inputs/tasks" / f"{task_id}.json", task_input)
        result_path = root / "tasks" / task_id
        row = {
            "task_id": task_id,
            "base_commit": revision,
            "result_path": str(result_path),
            "status": "PENDING",
        }
        if smoke and task_id == smoke["task_id"]:
            if smoke["base_commit"] != revision:
                raise ValueError("Smoke base revision differs")
            old_input = read_json(args.reuse_smoke / "inputs/spec-input.json")
            if any(old_input.get(key) != value for key, value in task_input.items()):
                raise ValueError("Smoke task input differs")
            shutil.copytree(args.reuse_smoke, result_path)
            row.update(
                status="A_COMPLETED",
                reused_from=str(args.reuse_smoke.resolve()),
                ledger_quality=smoke["ledger_quality"],
            )
            write_json(
                result_path / "reuse.json",
                {
                    "source": row["reused_from"],
                    "ledger_sha256": sha256_bytes((result_path / "v5/ledger.json").read_bytes()),
                },
            )
        else:
            repository = args.cache / task_id
            actual = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", revision + "^{commit}"], text=True
            ).strip()
            if actual != revision:
                raise ValueError(f"Cached base revision differs: {task_id}")
            destination = root / "baselines" / task_id
            info = export_tree(repository, revision, destination)
            row.update(
                repository=str(destination),
                **info,
                baseline_files_sha256=asyncio.run(
                    FrozenWorkspace(destination).tracked_diff_digest()
                ),
            )
        state["tasks"].append(row)
        write_json(root / "batch.json", state)
        print(f"Prepared {task_id}", flush=True)
    state["status"] = "READY"
    write_json(root / "batch.json", state)


def run(args):
    root = args.output.resolve()
    state = read_json(root / "batch.json")
    if state["status"] != "READY":
        raise ValueError("Only a fully prepared, unstarted batch can be launched")
    state.update(
        status="RUNNING", pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat()
    )
    write_json(root / "batch.json", state)
    with (root / "ssh-keepalive.stderr.log").open("wb") as ssh_log:
        # An active stdin channel prevents ControlPersist from expiring the existing tunnel.
        anchor = subprocess.Popen(
            [
                "ssh",
                "-S",
                str(args.ssh_socket),
                "-o",
                "BatchMode=yes",
                "-T",
                "-p",
                str(args.ssh_port),
                args.ssh_host,
                "cat >/dev/null",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=ssh_log,
        )
        power = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        state.update(ssh_keepalive_pid=anchor.pid, sleep_guard_pid=power.pid)
        try:
            time.sleep(0.3)
            for row in state["tasks"]:
                if row["status"] != "PENDING":
                    continue
                if anchor.poll() is not None:
                    raise RuntimeError(
                        "SSH keepalive ended; batch stopped before starting the next task"
                    )
                models = json.load(urlopen("http://127.0.0.1:3000/v1/models", timeout=15))
                if not any(m["id"] == "qwen35-9b" for m in models["data"]):
                    raise RuntimeError("The expected Qwen model is unavailable")
                if (
                    next(m["root"] for m in models["data"] if m["id"] == "qwen35-9b")
                    != state["model_root"]
                ):
                    raise RuntimeError("The served Qwen checkpoint changed during the batch")
                write_json(root / "inputs/runtime-models.json", models)
                row.update(status="RUNNING", started_at=datetime.now(timezone.utc).isoformat())
                state["current_task"] = row["task_id"]
                write_json(root / "batch.json", state)
                print(f"Starting {row['task_id']}", flush=True)
                command = [
                    sys.executable,
                    "-m",
                    "a_agent_finetune.run_e_inference",
                    "--repository",
                    row["repository"],
                    "--task-input",
                    str(root / "inputs/tasks" / f"{row['task_id']}.json"),
                    "--task-id",
                    row["task_id"],
                    "--base-commit",
                    row["base_commit"],
                    "--output",
                    row["result_path"],
                    "--timeout-sec",
                    str(state["timeout_sec"]),
                    "--max-tool-calls",
                    str(state["max_tool_calls"]),
                    "--loop-repeat-limit",
                    str(state.get("loop_repeat_limit", 8)),
                    "--loop-max-cycle-length",
                    str(state.get("loop_max_cycle_length", 8)),
                ]
                if not state.get("loop_detection", True):
                    command.append("--no-loop-detection")
                env = {**os.environ, "PYTHONPATH": str(root / "runtime"), "PYTHONUNBUFFERED": "1"}
                with (root / f"{row['task_id']}.log").open("wb") as log:
                    child = subprocess.Popen(
                        command, cwd=root / "runtime", env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    row["pid"] = child.pid
                    write_json(root / "batch.json", state)
                    exit_code = child.wait()
                manifest_path = Path(row["result_path"]) / "manifest.json"
                result = read_json(manifest_path) if manifest_path.exists() else {}
                row.update(
                    status=result.get("final_state", "PROCESS_FAILED"),
                    exit_code=exit_code,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    ledger_quality=result.get("ledger_quality"),
                    error=result.get("error"),
                )
                print(f"Finished {row['task_id']}: {row['status']}", flush=True)
                state["current_task"] = None
                write_json(root / "batch.json", state)
            state["status"] = "COMPLETED"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            state.update(status="STOPPED_INFRASTRUCTURE", error=str(error))
            print(f"Batch stopped: {error}", flush=True)
        finally:
            if anchor.stdin:
                anchor.stdin.close()
            try:
                anchor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                anchor.terminate()
            power.terminate()
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(root / "batch.json", state)
    return 0 if state["status"] == "COMPLETED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=HERE / "data/f_teacher_observed_30_v4")
    parser.add_argument("--cache", type=Path, default=HERE / ".cache/baselines")
    parser.add_argument("--reuse-smoke", type=Path, help="Optional result generated under the same conditions")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--max-tool-calls", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--loop-repeat-limit", type=int, default=8,
        help="Consecutive or periodic no-progress repetitions before stopping",
    )
    parser.add_argument(
        "--loop-max-cycle-length", type=int, default=8,
        help="Maximum action-cycle length checked by the loop detector",
    )
    parser.add_argument(
        "--no-loop-detection", dest="loop_detection", action="store_false",
        help="Disable online repeated-action detection",
    )
    parser.set_defaults(loop_detection=True)
    parser.add_argument("--ssh-socket", type=Path, default=Path("/tmp/e-qwen-autodl-47076.sock"))
    parser.add_argument("--ssh-host", default="root@connect.bjb1.seetacloud.com")
    parser.add_argument("--ssh-port", type=int, default=47076)
    args = parser.parse_args()
    if (
        args.timeout_sec < 1
        or args.max_tool_calls < 0
        or args.loop_repeat_limit < 2
        or args.loop_max_cycle_length < 1
    ):
        parser.error("timeout/budget/loop limits are invalid")
    if args.mode == "prepare":
        prepare(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
