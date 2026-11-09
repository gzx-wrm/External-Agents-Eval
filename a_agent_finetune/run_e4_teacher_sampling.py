"""Freeze and sample E4 teacher continuations. No student/GPU inference or training."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from a_agent_finetune.run_e_batch import export_tree
from a_agent_finetune.run_e_inference import FrozenWorkspace, output_validity
from external_agents_eval.agents.pier.pi_model_config import PiStageModelConfig

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
IMAGE = "external-agents-eval/pi-e-smoke:v0.84.2-arm64"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = read(args.seeds / "manifest.json")
    model = PiStageModelConfig.from_value(PROJECT / "configs/v6.1-deepseek-v4.1-flash.json")
    runtime = root / "runtime"
    shutil.copytree(PROJECT / "external_agents_eval", runtime / "external_agents_eval",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "a_agent_finetune").mkdir()
    for name in ("run_e4_teacher_sampling.py", "run_e_batch.py", "run_e_inference.py", "e4_teacher_continue.mjs"):
        shutil.copy2(HERE / name, runtime / "a_agent_finetune" / name)
    state = {"experiment": "E4", "status": "PREPARING", "created_at": now(),
             "concurrency": args.concurrency, "timeout_sec": 3600, "tool_call_limit": None,
             "teacher": model.default.to_metadata(), "thinking": "high",
             "image": IMAGE, "image_id": subprocess.check_output(
                 ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip(),
             "seed_pool": str(args.seeds.resolve()), "training_started": False, "samples": []}
    write(root / "seed-manifest.json", manifest)
    for row in manifest["samples"]:
        source = args.seeds / row["seed_file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["seed_sha256"]:
            raise ValueError(f"Seed hash differs: {row['sample_id']}")
        baseline = root / "baselines" / row["task_id"]
        if not baseline.exists():
            repo = args.cache / row["task_id"]
            actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse",
                                              row["base_commit"] + "^{commit}"], text=True).strip()
            if actual != row["base_commit"]:
                raise ValueError(f"Base revision differs: {row['task_id']}")
            export_tree(repo, row["base_commit"], baseline)
        directory = root / "samples" / row["sample_id"]
        directory.mkdir(parents=True)
        shutil.copy2(source, directory / "seed.json")
        shutil.copy2(runtime / "a_agent_finetune/e4_teacher_continue.mjs", directory / "continue.mjs")
        write(directory / "models.json", model.models_json(model.default))
        write(directory / "auth.json", {})
        write(directory / "config.json", {"experiment": "E4", "model": model.default.model,
              "timeout_sec": 3600, "max_output_tokens": model.default.model_metadata["maxTokens"],
              "seed_sha256": row["seed_sha256"], "base_commit": row["base_commit"],
              "additional_user_instructions": False, "manual_ledger_in_teacher_input": False})
        state["samples"].append({"sample_id": row["sample_id"], "task_id": row["task_id"],
                                 "status": "PENDING", "seed_sha256": row["seed_sha256"],
                                 "baseline_files_sha256_before": asyncio.run(
                                     FrozenWorkspace(baseline).tracked_diff_digest())})
        print(f"Prepared {row['sample_id']}", flush=True)
    digest = hashlib.sha256()
    for p in sorted(runtime.rglob("*")):
        if p.is_file():
            digest.update(p.relative_to(runtime).as_posix().encode() + b"\0" + p.read_bytes())
    state.update(status="READY", runtime_sha256=digest.hexdigest())
    write(root / "batch.json", state)


async def sample(root, row, state, semaphore, dry_run=False):
    async with semaphore:
        directory = root / "samples" / row["sample_id"]
        baseline = root / "baselines" / row["task_id"]
        if dry_run:
            directory = root / "dry-run" / row["sample_id"]
            shutil.copytree(root / "samples" / row["sample_id"], directory)
        name = "e4-teacher-" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
        command = ["docker", "run", "--rm", "--name", name, "--entrypoint", "/usr/bin/timeout",
                   "--workdir", "/work/base", "--volume", f"{baseline}:/work/base:ro",
                   "--volume", f"{directory}:/work/output", "--env", "DEEPSEEK_API_KEY",
                   "--env", "PI_OFFLINE=1", "--env", "PI_TELEMETRY=0", "--env", "DO_NOT_TRACK=1",
                   "--env", "PI_CODING_AGENT_DIR=/work/output", IMAGE,
                   "--signal=TERM", "--kill-after=30s", "3610s",
                   "/opt/pi-runtime/bin/node", "/work/output/continue.mjs"]
        if dry_run:
            command.append("--dry-run")
        row.update(status="RUNNING", started_at=now(), container=name)
        if not dry_run:
            write(root / "batch.json", state)
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            process = await asyncio.create_subprocess_exec(*command, stdout=out, stderr=err)
            code = await process.wait()
        result = read(directory / "status.json") if (directory / "status.json").exists() else {
            "status": "FAILED", "error": "Runtime exited without status"}
        result["exit_code"] = code
        result["baseline_files_sha256_after"] = await FrozenWorkspace(baseline).tracked_diff_digest()
        result["baseline_unchanged"] = result["baseline_files_sha256_after"] == row["baseline_files_sha256_before"]
        raw = (directory / "output.txt").read_text() if (directory / "output.txt").exists() else ""
        if not dry_run:
            result.update(output_validity(raw))
            # Schema success is only an automated screen; semantic review still required.
            if not result.get("schema_valid") or not result["baseline_unchanged"]:
                result["status"] = "REVIEW_REQUIRED_INVALID_OUTPUT" if code == 0 else "FAILED"
            result["trainable"] = False
        write(directory / "result.json", result)
        if not dry_run:
            row.update(status=result["status"], finished_at=now(), exit_code=code,
                       schema_valid=result.get("schema_valid", False),
                       assistant_turns=result.get("assistant_turns", 0), tool_calls=result.get("tool_calls", 0),
                       usage=result.get("usage"))
            write(root / "batch.json", state)
        print(json.dumps({"sample_id": row["sample_id"], "status": result["status"],
                          "exit_code": code, "dry_run": dry_run}), flush=True)
        return result


async def run(args):
    root = args.output.resolve()
    state = read(root / "batch.json")
    if args.mode == "check":
        result = await sample(root, dict(state["samples"][0]), state, asyncio.Semaphore(1), True)
        if result["exit_code"] != 0 or not result.get("prefix_verified") or result.get("tool_calls") != 1:
            raise RuntimeError("Offline prefix/tool continuation check failed")
        return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY is required")
    state.update(status="RUNNING", started_at=now(), supervisor_pid=os.getpid())
    write(root / "batch.json", state)
    semaphore = asyncio.Semaphore(state["concurrency"])
    outcomes = await asyncio.gather(*(sample(root, row, state, semaphore)
                                     for row in state["samples"] if row["status"] == "PENDING"),
                                    return_exceptions=True)
    errors = [str(x) for x in outcomes if isinstance(x, BaseException)]
    state.update(status="SAMPLING_FINISHED_REVIEW_PENDING" if not errors else "SAMPLING_FAILED",
                 finished_at=now(), errors=errors)
    write(root / "batch.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, default=HERE / "data/e_recovery_pilot_v1")
    parser.add_argument("--cache", type=Path, default=HERE / ".cache/baselines")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
