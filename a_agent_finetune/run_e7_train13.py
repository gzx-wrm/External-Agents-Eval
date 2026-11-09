"""Run the E7 A-only training-set evaluation with online loop detection."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = HERE / "runs/e7-train13-20261012-r1"
SOURCE = HERE / "runs/base-e6-train13-20261011-r1"
TASKS = [
    "aiomonitor-task-snapshots-diff",
    "expr-try-catch-errors",
    "testem-per-launcher-reports",
    "yaegi-go-embed-directives",
    "boa-hierarchical-evaluation-cancellation",
    "vulture-persistent-analysis-cache",
    "obsidian-linter-link-format-conversion",
    "obsidian-linter-auto-table-of-contents",
    "adaptix-name-mapping-aliases",
    "anko-typed-variable-bindings",
    "arktype-json-schema-refs-dependencies",
    "langchain-request-coalescing",
    "termenv-preserve-ansi-resets",
]
MODEL = "qwen35-9b-e7"
IMAGE = "external-agents-eval/pi-e-smoke:v0.84.2-arm64"


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def prepare() -> None:
    if RUN.exists():
        raise SystemExit(f"Output already exists: {RUN}")
    (RUN / "inputs/tasks").mkdir(parents=True)
    (RUN / "tasks").mkdir()
    for name in ("system-prompt.txt", "inference_config.json"):
        (RUN / "inputs" / name).write_bytes((SOURCE / "inputs" / name).read_bytes())
    config = json.loads((SOURCE / "inputs/e6-stage_model_config.json").read_text())
    for raw in [config["default"], config["A"], config["json_corrector"]]:
        raw["model"] = MODEL
    config["default"]["model_metadata"]["name"] = "Qwen3.5-9B E7 LoRA"
    write(RUN / "inputs/stage_model_config.json", config)
    write(RUN / "inputs/controller_config.json", {
        "model": MODEL, "max_format_retries": 1,
        "max_semantic_retries": 0, "no_compaction": True,
    })
    task_meta = {}
    for task in TASKS:
        src = SOURCE / "inputs/tasks" / f"{task}.json"
        dst = RUN / "inputs/tasks" / f"{task}.json"
        dst.write_bytes(src.read_bytes())
        task_meta[task] = {
            "base_commit": json.loads(dst.read_text())["base_commit"]
            if "base_commit" in json.loads(dst.read_text())
            else None,
            "repository": str(ROOT / "a_agent_finetune/g1/data/repositories" / task),
        }
    # Existing task inputs omit base_commit; use the frozen G1 task manifest.
    for line in (HERE / "g1/data/tasks.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["task_id"] in task_meta:
            task_meta[row["task_id"]]["base_commit"] = row["base_commit"]
    if any(not row["base_commit"] for row in task_meta.values()):
        raise ValueError("Missing base commit")
    state = {
        "experiment": "E7 A-only training-set evaluation",
        "status": "READY",
        "model": MODEL,
        "loop_detection": {"enabled": True, "repeat_limit": 8,
                            "max_cycle_length": 8, "fingerprint": "action-and-result"},
        "sampling": json.loads((RUN / "inputs/inference_config.json").read_text()),
        "tasks": TASKS,
        "task_meta": task_meta,
        "completed": 0,
        "results": [],
    }
    write(RUN / "batch.json", state)


def summarize(state: dict) -> None:
    rows = []
    for task in TASKS:
        manifest = RUN / "tasks" / task / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        inv = (data.get("invocations") or [{}])[0]
        rows.append({
            "task_id": task, "final_state": data.get("final_state"),
            "error": data.get("error"), "loop_detected": inv.get("loop_detected", False),
            "loop_report": inv.get("loop_report"), "model_calls": inv.get("model_calls", 0),
            "tool_calls": inv.get("tool_calls", 0), "tool_errors": inv.get("tool_errors", 0),
            "input_tokens": inv.get("input_tokens", 0), "output_tokens": inv.get("output_tokens", 0),
            "elapsed_sec": inv.get("elapsed_sec", 0),
            "termination_reason": inv.get("termination_reason"),
        })
    write(RUN / "comparison.json", rows)
    lines = ["# E7 training-set A-only evaluation", "",
             "| Task | State | Loop | Model calls | Tool calls | Minutes |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['task_id']} | {row['final_state']} | "
                     f"{'yes' if row['loop_detected'] else 'no'} | {row['model_calls']} | "
                     f"{row['tool_calls']}/{row['tool_errors']} | {row['elapsed_sec']/60:.1f} |")
    lines.append("")
    lines.append(f"Completed: {sum(r['final_state'] == 'A_COMPLETED' for r in rows)}/{len(rows)}; "
                 f"loop detected: {sum(r['loop_detected'] for r in rows)}.")
    (RUN / "comparison.md").write_text("\n".join(lines) + "\n")


def run() -> None:
    state = json.loads((RUN / "batch.json").read_text())
    state.update(status="RUNNING", started_at=time.time(), current_task=None)
    write(RUN / "batch.json", state)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", OPENAI_API_KEY="LOCAL_E7")
    for task in TASKS:
        out = RUN / "tasks" / task
        log_path = RUN / "tasks" / f"{task}.log"
        repo = ROOT / "a_agent_finetune/g1/data/repositories" / task
        meta = state["task_meta"][task]
        cmd = [sys.executable, "-B", "-u", "-m", "a_agent_finetune.run_e_inference",
               "--repository", str(repo),
               "--task-input", str(RUN / "inputs/tasks" / f"{task}.json"),
               "--task-id", task, "--base-commit", meta["base_commit"],
               "--output", str(out), "--timeout-sec", "3600", "--max-tool-calls", "0",
               "--image", IMAGE, "--system-prompt", str(RUN / "inputs/system-prompt.txt"),
               "--stage-model-config", str(RUN / "inputs/stage_model_config.json"),
               "--controller-config", str(RUN / "inputs/controller_config.json"),
               "--inference-config", str(RUN / "inputs/inference_config.json"),
               "--loop-repeat-limit", "8", "--loop-max-cycle-length", "8"]
        state["current_task"] = task
        write(RUN / "batch.json", state)
        started = time.time()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest = out / "manifest.json"
        result = json.loads(manifest.read_text()) if manifest.exists() else {
            "final_state": "PROCESS_FAILED", "error": f"exit={proc.returncode}"
        }
        state["results"].append({"task_id": task, "final_state": result.get("final_state"),
                                "error": result.get("error"), "elapsed_sec": time.time() - started})
        state["completed"] += 1
        state["last_task"] = task
        summarize(state)
        write(RUN / "batch.json", state)
    state.update(status="COMPLETED", current_task=None, finished_at=time.time())
    summarize(state)
    write(RUN / "batch.json", state)


if __name__ == "__main__":
    prepare()
    run()
