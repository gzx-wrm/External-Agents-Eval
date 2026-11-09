"""Run E8: original Qwen3.5-9B base with an explicit tool-repeat guard."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = HERE / "runs/e8-base-train13-20261012-r1"
SOURCE = HERE / "runs/base-e6-train13-20261011-r1"
IMAGE = "external-agents-eval/pi-e-smoke:v0.84.2-arm64"
MODEL = "qwen35-9b-e8-base"
REMOTE_ROOT = "/root/autodl-tmp/e8-base-train13-20261012-r1"
REMOTE_MODEL = "/root/autodl-tmp/g1/models/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SSH_HOST = "root@connect.nma1.seetacloud.com"
SSH_PORT = "57563"
VLLM = "/root/autodl-tmp/g1/venv/bin/vllm"
CACHE_ROOT = "/root/autodl-tmp/e6"
SOCKET = Path("/tmp/e8-base-train13-20261012.sock")
LOCAL_PORT = 3000
# Inference-time budget gates; the context window matches the vLLM --max-model-len.
# file_segment_limit stays 0: the teacher itself reads one file 10 times at the
# median (36/60 training rows exceed 8), so a segment cap blocks normal work.
BUDGET = {
    "context_window": 172032, "soft_fraction": 0.50, "hard_fraction": 0.75,
    "file_segment_limit": 0, "loop_action_repeat_limit": 16, "gate_block_limit": 6,
    "unknown_tool_turn_limit": 8,
}
BUDGET_ENABLED = False
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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def ssh_base() -> list[str]:
    return [
        "ssh",
        "-S",
        str(SOCKET),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        SSH_PORT,
        SSH_HOST,
    ]


def remote(command: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        ssh_base() + [command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"remote command failed ({proc.returncode}): {proc.stdout[-2000:]}")
    return proc.stdout


def establish_control_master() -> None:
    SOCKET.unlink(missing_ok=True)
    password = os.environ.get("E8_SSH_PASSWORD")
    if not password:
        raise RuntimeError("E8_SSH_PASSWORD is required for the requested A800 instance")
    script = r'''
set timeout 40
spawn ssh -M -S $env(E8_SOCK) -o ControlPersist=12h -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -fN -p $env(E8_PORT) $env(E8_HOST)
expect {
  -re "(?i)password:" { send "$env(E8_SSH_PASSWORD)\r"; exp_continue }
  eof {}
  timeout { exit 124 }
}
'''
    env = dict(os.environ, E8_SOCK=str(SOCKET), E8_PORT=SSH_PORT, E8_HOST=SSH_HOST)
    proc = subprocess.run(["expect", "-c", script], env=env, text=True)
    if proc.returncode:
        raise RuntimeError(f"could not establish SSH control master (exit={proc.returncode})")
    remote("echo ssh-control-ready", timeout=30)


def upload(path: str, data: bytes) -> None:
    proc = subprocess.run(
        ssh_base() + [f"cat > {shlex.quote(path)}"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"upload failed: {proc.stdout.decode(errors='replace')[-2000:]}")


def start_remote_service() -> None:
    start = f'''#!/usr/bin/env bash
set -euo pipefail
ROOT={shlex.quote(REMOTE_ROOT)}
MODEL={shlex.quote(REMOTE_MODEL)}
mkdir -p "$ROOT"
if test -f "$ROOT/server.pid"; then kill "$(cat "$ROOT/server.pid")" 2>/dev/null || true; fi
nohup env HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \\
  VLLM_CACHE_ROOT={shlex.quote(CACHE_ROOT)}/vllm-cache \\
  XDG_CACHE_HOME={shlex.quote(CACHE_ROOT)}/cache \\
  VLLM_USE_FLASHINFER_SAMPLER=0 \\
  {shlex.quote(VLLM)} serve "$MODEL" \\
  --tokenizer "$MODEL" \\
  --served-model-name {shlex.quote(MODEL)} \\
  --dtype bfloat16 \\
  --language-model-only \\
  --max-model-len 172032 \\
  --max-num-seqs 1 \\
  --max-num-batched-tokens 2048 \\
  --gpu-memory-utilization 0.90 \\
  --enable-prefix-caching \\
  --mamba-cache-mode all \\
  --mamba-block-size 1024 \\
  --enforce-eager \\
  --generation-config vllm \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder \\
  --reasoning-parser qwen3 \\
  --default-chat-template-kwargs '{{"enable_thinking":false}}' \\
  --host 127.0.0.1 --port 3000 >"$ROOT/server.log" 2>&1 </dev/null &
echo $! > "$ROOT/server.pid"
'''
    remote(f"rm -rf {shlex.quote(REMOTE_ROOT)} && mkdir -p {shlex.quote(REMOTE_ROOT)}", timeout=60)
    upload(REMOTE_ROOT + "/start.sh", start.encode())
    remote(f"chmod +x {shlex.quote(REMOTE_ROOT + '/start.sh')} && bash {shlex.quote(REMOTE_ROOT + '/start.sh')}", timeout=60)


def wait_model(deadline: int = 1800) -> None:
    url = f"http://127.0.0.1:{LOCAL_PORT}/v1/models"
    end = time.time() + deadline
    last = ""
    while time.time() < end:
        try:
            data = json.load(urlopen(url, timeout=10))
            ids = [row.get("id") for row in data.get("data", [])]
            if MODEL in ids:
                write_json(RUN / "service-models.json", data)
                return
            last = f"models={ids}"
        except Exception as error:  # service startup is expected to take time
            last = repr(error)
        time.sleep(5)
    raise RuntimeError(f"model service did not become ready: {last}")


def make_inputs() -> dict[str, str]:
    if RUN.exists():
        raise RuntimeError(f"output already exists: {RUN}")
    (RUN / "inputs/tasks").mkdir(parents=True)
    (RUN / "tasks").mkdir()
    (RUN / "inputs/system-prompt.txt").write_bytes(
        (HERE / "e8_prompt_repetition_guard.txt").read_bytes()
    )
    (RUN / "inputs/inference_config.json").write_bytes(
        (SOURCE / "inputs/inference_config.json").read_bytes()
    )
    stage = json.loads((SOURCE / "inputs/base-stage_model_config.json").read_text())
    for section in ("default", "A", "json_corrector"):
        stage[section]["model"] = MODEL
    stage["default"]["model_metadata"]["name"] = "Qwen3.5-9B original base with E8 repeat guard"
    write_json(RUN / "inputs/stage_model_config.json", stage)
    controller = json.loads((SOURCE / "inputs/base-controller_config.json").read_text())
    controller["model"] = MODEL
    write_json(RUN / "inputs/controller_config.json", controller)

    task_meta: dict[str, str] = {}
    all_tasks = {}
    sources = [HERE / "g1/data/tasks.jsonl"]
    extra = HERE / "g1/data/tasks-extra.jsonl"
    if extra.exists():
        sources.append(extra)
    for source in sources:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            all_tasks[row["task_id"]] = row
    for task in TASKS:
        source = SOURCE / "inputs/tasks" / f"{task}.json"
        target = RUN / "inputs/tasks" / f"{task}.json"
        target.write_bytes(source.read_bytes())
        task_meta[task] = all_tasks[task]["base_commit"]
    state = {
        "experiment": "E8 original Qwen3.5-9B base with explicit tool-repeat guard",
        "status": "READY",
        "model": MODEL,
        "prompt_path": str((RUN / "inputs/system-prompt.txt").resolve()),
        "prompt_sha256": __import__("hashlib").sha256((RUN / "inputs/system-prompt.txt").read_bytes()).hexdigest(),
        "baseline_prompt_sha256": __import__("hashlib").sha256((SOURCE / "inputs/system-prompt.txt").read_bytes()).hexdigest(),
        "prompt_delta": "one repetition/recovery rule block inserted; all other baseline prompt bytes copied",
        "loop_detection": {"enabled": True, "repeat_limit": 8, "max_cycle_length": 8, "fingerprint": "action-and-result"},
        "budget_control": BUDGET if BUDGET_ENABLED else None,
        "sampling": json.loads((RUN / "inputs/inference_config.json").read_text()),
        "tasks": TASKS,
        "task_meta": {task: {"base_commit": commit} for task, commit in task_meta.items()},
        "completed": 0,
        "results": [],
    }
    write_json(RUN / "batch.json", state)
    return state


def summarize(state: dict) -> None:
    rows = []
    for task in TASKS:
        manifest = RUN / "tasks" / task / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        invocation = (data.get("invocations") or [{}])[0]
        rows.append({
            "task_id": task,
            "final_state": data.get("final_state"),
            "error": data.get("error"),
            "loop_detected": invocation.get("loop_detected", False),
            "loop_report": invocation.get("loop_report"),
            "model_calls": invocation.get("model_calls", 0),
            "tool_calls": invocation.get("tool_calls", 0),
            "tool_errors": invocation.get("tool_errors", 0),
            "input_tokens": invocation.get("input_tokens", 0),
            "output_tokens": invocation.get("output_tokens", 0),
            "elapsed_sec": invocation.get("elapsed_sec", 0),
            "termination_reason": invocation.get("termination_reason"),
        })
    write_json(RUN / "comparison.json", rows)
    lines = ["# E8 original base training-set evaluation", "", "| Task | State | Loop | Model calls | Tool calls | Minutes |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['final_state']} | {'yes' if row['loop_detected'] else 'no'} | "
            f"{row['model_calls']} | {row['tool_calls']}/{row['tool_errors']} | {row['elapsed_sec']/60:.1f} |"
        )
    lines.extend([
        "",
        f"Completed: {sum(row['final_state'] == 'A_COMPLETED' for row in rows)}/{len(rows)}; "
        f"loop detected: {sum(row['loop_detected'] for row in rows)}.",
    ])
    (RUN / "comparison.md").write_text("\n".join(lines) + "\n")


def run_tasks(state: dict) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", OPENAI_API_KEY="LOCAL_E8")
    for task in TASKS:
        output = RUN / "tasks" / task
        repository = HERE / "g1/data/repositories" / task
        command = [
            sys.executable, "-B", "-u", "-m", "a_agent_finetune.run_e_inference",
            "--repository", str(repository),
            "--task-input", str(RUN / "inputs/tasks" / f"{task}.json"),
            "--task-id", task,
            "--base-commit", state["task_meta"][task]["base_commit"],
            "--output", str(output),
            "--timeout-sec", "3600",
            "--max-tool-calls", "0",
            "--image", IMAGE,
            "--system-prompt", str(RUN / "inputs/system-prompt.txt"),
            "--stage-model-config", str(RUN / "inputs/stage_model_config.json"),
            "--controller-config", str(RUN / "inputs/controller_config.json"),
            "--inference-config", str(RUN / "inputs/inference_config.json"),
            "--loop-repeat-limit", "8",
            "--loop-max-cycle-length", "8",
        ]
        if BUDGET_ENABLED:
            command += [
                "--context-window", str(BUDGET["context_window"]),
                "--budget-soft-fraction", str(BUDGET["soft_fraction"]),
                "--budget-hard-fraction", str(BUDGET["hard_fraction"]),
                "--file-segment-limit", str(BUDGET["file_segment_limit"]),
                "--loop-action-repeat-limit", str(BUDGET["loop_action_repeat_limit"]),
                "--gate-block-limit", str(BUDGET["gate_block_limit"]),
                "--unknown-tool-turn-limit", str(BUDGET["unknown_tool_turn_limit"]),
            ]
        state["current_task"] = task
        write_json(RUN / "batch.json", state)
        started = time.time()
        with (RUN / "tasks" / f"{task}.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest = output / "manifest.json"
        result = json.loads(manifest.read_text()) if manifest.exists() else {"final_state": "PROCESS_FAILED", "error": f"exit={process.returncode}"}
        state["results"].append({"task_id": task, "final_state": result.get("final_state"), "error": result.get("error"), "elapsed_sec": time.time() - started})
        state["completed"] += 1
        state["last_task"] = task
        summarize(state)
        write_json(RUN / "batch.json", state)


def main() -> None:
    state = make_inputs()
    state.update(status="STARTING", started_at=time.time())
    write_json(RUN / "batch.json", state)
    tunnel = None
    try:
        establish_control_master()
        start_remote_service()
        tunnel = subprocess.Popen(
            ssh_base() + ["-L", f"127.0.0.1:{LOCAL_PORT}:127.0.0.1:3000", "-N"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=(RUN / "ssh-tunnel.log").open("w"),
            start_new_session=True,
        )
        state["status"] = "RUNNING"
        state["tunnel_pid"] = tunnel.pid
        write_json(RUN / "batch.json", state)
        wait_model()
        run_tasks(state)
        state.update(status="COMPLETED", current_task=None, finished_at=time.time())
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}", finished_at=time.time())
        raise
    finally:
        summarize(state)
        write_json(RUN / "batch.json", state)
        try:
            remote(f"if test -f {shlex.quote(REMOTE_ROOT + '/server.pid')}; then kill \"$(cat {shlex.quote(REMOTE_ROOT + '/server.pid')})\" 2>/dev/null || true; fi", timeout=30)
        except Exception:
            pass
        if tunnel is not None:
            try:
                tunnel.terminate()
                tunnel.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        try:
            subprocess.run(
                [
                    "ssh", "-S", str(SOCKET), "-o", "BatchMode=yes", "-O", "exit",
                    "-p", SSH_PORT, SSH_HOST,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
        except Exception:
            pass
        SOCKET.unlink(missing_ok=True)
        write_json(RUN / "batch.json", state)


if __name__ == "__main__":
    main()
