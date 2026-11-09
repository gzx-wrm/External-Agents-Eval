"""Native Mini + DeepSeek on the exact FT30; each slot solves then grades."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import shlex
import tarfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import minisweagent
import yaml
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.environments import get_environment
from minisweagent.models import get_model

import docker


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if secret := os.environ.get("DEEPSEEK_API_KEY"):
        text = text.replace(secret, "[REDACTED]")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def upload(container, data, destination, *, mode=0o644):
    if isinstance(data, Path):
        data = data.read_bytes()
    parent, name = destination.rsplit("/", 1)
    prepared = container.exec_run(["mkdir", "-p", parent])
    if prepared.exit_code:
        raise RuntimeError("Cannot create container artifact directory")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        item = tarfile.TarInfo(name)
        item.size, item.mode = len(data), mode
        archive.addfile(item, io.BytesIO(data))
    if not container.put_archive(parent, stream.getvalue()):
        raise RuntimeError("Container artifact upload failed")


def snapshot_command(base):
    base = shlex.quote(base)
    return (
        'snapshot_dir="$(mktemp -d)" && '
        "trap 'rm -rf -- \"$snapshot_dir\"' EXIT && "
        'snapshot_index="$snapshot_dir/index" && '
        f'GIT_INDEX_FILE="$snapshot_index" git read-tree {base} && '
        'GIT_INDEX_FILE="$snapshot_index" git add -A -- . && '
        f'GIT_INDEX_FILE="$snapshot_index" git diff --cached --binary --full-index --no-ext-diff {base} -- .'
    )


def checked(environment, command, *, timeout=300):
    result = environment.execute({"command": command}, timeout=timeout)
    if result["returncode"]:
        raise RuntimeError("Preparation/artifact command failed: " + result["output"][-2500:])
    return result["output"]


def load_tasks(root):
    tasks = json.loads((root / "manifests/public-tasks.json").read_text())
    suite = json.loads((root / "manifests/eval30_current.json").read_text())
    if len(tasks) != 30 or [task["task_id"] for task in tasks] != suite["task_ids"]:
        raise ValueError("Task list differs from the fixed FT30")
    if "sqlite-utils-safe-import-checkpoints" in suite["task_ids"]:
        raise ValueError("The permanently excluded SQLite task was reintroduced")
    expected = json.loads((root / "manifests/task-file-sha256.json").read_text())
    for name, digest in expected.items():
        if hashlib.sha256((root / "grader" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen task file changed: " + name)
    return tasks


def prepare_config(root):
    source = Path(minisweagent.__file__).parent / "config/mini.yaml"
    upstream = yaml.safe_load(source.read_text())
    config = copy.deepcopy(upstream)
    config["agent"] = {k: v for k, v in config["agent"].items() if k in AgentConfig.model_fields}
    config["agent"].update(step_limit=0, cost_limit=0, wall_time_limit_seconds=10800)
    config["environment"].update(
        environment_class="docker", cwd="/app", forward_env=[], container_timeout="4h",
        run_args=["--rm", "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                  "--pids-limit=512"],
    )
    config["model"].update(
        model_name="deepseek/deepseek-flash", cost_tracking="ignore_errors",
        model_class="native_compat.DeepSeekNativeModel",
    )
    config["model"]["model_kwargs"].update(
        api_base="https://api.deepseek.com/v1", reasoning_effort="high", max_tokens=65536,
        extra_body={"thinking": {"type": "enabled"}},
    )
    destination = root / "configs/native-deepseek-v4.1-flash.yaml"
    if destination.exists() and yaml.safe_load(destination.read_text()) != config:
        raise ValueError("Existing experiment configuration differs; use a new reviewed run")
    destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    (root / "configs/upstream-mini.yaml").write_bytes(source.read_bytes())
    return config


def prepare_repository(root, environment, container, task, output):
    # Preserve deterministic image-only changes, as the FT workflow does.
    environment_patch = checked(environment, snapshot_command(task["base_commit"]))
    (output / "environment-baseline.patch").write_text(environment_patch)
    upload(container, root / "scripts/sanitize_repository.py", "/tmp/native_sanitize_repository.py")
    raw = checked(environment, "python /tmp/native_sanitize_repository.py --repository /app --revision "
                  + shlex.quote(task["base_commit"]))
    audit = json.loads(raw.splitlines()[-1])
    checked(environment, "git clean -fd -- .")
    if environment_patch:
        upload(container, environment_patch.encode(), "/tmp/native_environment.patch")
        checked(environment, "git apply --binary /tmp/native_environment.patch && rm /tmp/native_environment.patch")
    checked(environment, "git config core.hooksPath /dev/null && git branch main HEAD")
    restored = checked(environment, snapshot_command(audit["synthetic_commit"]))
    if restored != environment_patch:
        raise RuntimeError("Git isolation changed the image's runnable baseline")
    scan = checked(environment, "find /app /root /opt -name .git -type d -prune 2>/dev/null || true")
    others = [value for value in scan.splitlines() if value != "/app/.git"]
    for directory in others:
        checked(environment, "rm -rf -- " + shlex.quote(directory))
    check = checked(environment, "python - <<'PY'\nimport socket,os,json\ns=socket.socket();s.settimeout(2)\ntry:s.connect(('1.1.1.1',443));blocked=False\nexcept OSError:blocked=True\nfinally:s.close()\nprint(json.dumps({'blocked':blocked,'hidden_tests_present':os.path.exists('/tests/grader.py'),'solution_present':os.path.exists('/solution/solution.patch')}))\nPY")
    probe = json.loads(check.splitlines()[-1])
    container.reload()
    host = container.attrs["HostConfig"]
    if (not probe["blocked"] or probe["hidden_tests_present"] or probe["solution_present"]
            or host["NetworkMode"] != "none" or host.get("Binds") or host.get("Privileged")
            or environment.config.forward_env):
        raise RuntimeError("Native generation isolation audit failed")
    audit.update(network="none", hidden_tests_present=False, solution_present=False,
                 other_git_directories_removed=len(others),
                 environment_patch_sha256=hashlib.sha256(environment_patch.encode()).hexdigest(),
                 environment_patch_preserved=True, main_alias="synthetic baseline only",
                 cpus=host["NanoCpus"] / 1_000_000_000, memory_bytes=host["Memory"],
                 host_mounts=host.get("Binds") or [], forwarded_host_environment=[])
    save(output / "isolation.json", audit)
    return audit


def make_agent(config, environment, output):
    return DefaultAgent(get_model(config=config["model"]), environment,
                        **config["agent"], output_path=output / "native.traj.json")


def offline_probe(config, environment, output, task):
    import litellm

    requests = []

    def completion(**kwargs):
        requests.append(kwargs)
        assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["bash"]
        if len(requests) == 2:
            previous = [m for m in kwargs["messages"] if m["role"] == "assistant"][-1]
            assert previous["reasoning_content"] == " "
        command = ("printf 'native offline probe\\n' > native_offline_probe.txt" if len(requests) == 1
                   else "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Offline protocol probe.", "reasoning_content": "",
                "tool_calls": [{"id": "offline-" + str(len(requests)), "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
            },
        }], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    agent = make_agent(config, environment, output)
    with patch("litellm.completion", completion):
        info = agent.run(task["instruction"])
    assert info["exit_status"] == "Submitted"
    assert next(m for m in agent.messages if m["role"] == "assistant")["reasoning_content"] == ""
    return {"native_agent": type(agent).__module__ + "." + type(agent).__name__,
            "native_environment": type(environment).__module__ + "." + type(environment).__name__,
            "simulated_model_responses": len(requests), "model_api_calls": 0,
            "raw_reasoning_preserved": True, "outbound_compatibility_verified": True}


def grade(root, task, image_id, model_patch, output):
    """Run unchanged official tests in a fresh task image; no derived images."""
    output.mkdir(parents=True, exist_ok=True)
    client = docker.from_env(timeout=task["verifier_timeout_sec"] + 300)
    container = None
    try:
        container = client.containers.create(
            image_id, name="dsm-grade-" + uuid.uuid4().hex[:12], entrypoint=[],
            command=["sleep", "infinity"], working_dir="/app", network_mode="none",
            nano_cpus=int(task["cpus"]) * 1_000_000_000, mem_limit=str(task["memory_mb"]) + "m",
            labels={"external_eval": "native-mini-ft30-grading", "task": task["task_id"]},
        )
        container.start()
        tests = root / "grader/tasks" / task["task_id"] / "tests"
        # All 30 official verifier Dockerfiles contain exactly these four COPYs
        # and chmod. Inject the same bytes without building redundant images.
        for name in ["test.sh", "test.patch", "grader.py", "config.json"]:
            upload(container, tests / name, "/tests/" + name, mode=0o755 if name == "test.sh" else 0o644)
        upload(container, model_patch.encode(), "/logs/artifacts/model.patch")
        seconds = str(int(task["verifier_timeout_sec"]))
        execution = container.exec_run(
            ["timeout", "--signal=TERM", "--kill-after=15", seconds, "bash", "/tests/test.sh"],
            workdir="/app", stream=True,
        )
        with (output / "test-stdout.txt").open("wb") as log:
            for chunk in execution.output:
                log.write(chunk)
                log.flush()
        try:
            chunks, _ = container.get_archive("/logs/verifier")
            with tarfile.open(fileobj=io.BytesIO(b"".join(chunks))) as archive:
                archive.extractall(output, filter="data")
        except docker.errors.NotFound:
            pass  # Missing reward remains an explicit infrastructure failure.
        reward_path = output / "verifier/reward.json"
        if reward_path.exists():
            official = json.loads(reward_path.read_text())
            if official.get("reward") in [0, 1]:
                result = {"status": "scored", "reward": official["reward"], "official": official,
                          "task_id": task["task_id"], "task_checksum": task["task_checksum"]}
                save(output / "evaluation.json", result)
                return result
        result = {"status": "verifier_infrastructure_error", "task_id": task["task_id"],
                  "reward": None, "reason": "Official verifier produced no valid reward.json"}
        save(output / "evaluation.json", result)
        return result
    finally:
        if container is not None:
            container.remove(force=True)
        client.close()


def generate(task, root, output, config, image, mode):
    output.mkdir(parents=True, exist_ok=True)
    options = copy.deepcopy(config["environment"])
    options["image"] = image.id
    options["run_args"] += ["--cpus=" + str(task["cpus"]), "--memory=" + str(task["memory_mb"]) + "m"]
    environment = get_environment(options)
    client = docker.from_env()
    try:
        container = client.containers.get(environment.container_id)
        audit = prepare_repository(root, environment, container, task, output)
        if mode == "offline":
            info = offline_probe(config, environment, output, task)
        else:
            info = make_agent(config, environment, output).run(task["instruction"])
        model_patch = checked(environment, snapshot_command(audit["synthetic_commit"]))
        if mode == "offline" and "native_offline_probe.txt" not in model_patch:
            raise RuntimeError("Native untracked-file export failed")
        (output / "model.patch").write_text(model_patch)
        save(output / "agent-info.json", info)
        save(output / "prediction.json", {"task_id": task["task_id"], "task_checksum": task["task_checksum"],
                                           "model": "deepseek-flash", "model_patch": model_patch,
                                           "exit_status": info.get("exit_status")})
        return model_patch, info
    finally:
        if environment.container_id:
            try:
                client.containers.get(environment.container_id).remove(force=True)
            except docker.errors.NotFound:
                pass
            environment.container_id = None
        client.close()


def get_image(client, task, output):
    try:
        image, added = client.images.get(task["image"]), False
    except docker.errors.ImageNotFound:
        image, added = client.images.pull(task["image"]), True
    save(output / "image.json", {"name": task["image"], "id": image.id,
                                 "repo_digests": image.attrs.get("RepoDigests", []), "newly_pulled": added})
    return image, added


def preflight(root, task, config):
    output = root / "preflight" / task["task_id"]
    output.mkdir(parents=True, exist_ok=True)
    client = docker.from_env(timeout=1800)
    try:
        image, _ = get_image(client, task, output)
        _, offline = generate(task, root, output / "offline", config, image, "offline")
        gold_patch = (root / "grader/tasks" / task["task_id"] / "solution/solution.patch").read_text()
        gold = grade(root, task, image.id, gold_patch, output / "gold")
        control = grade(root, task, image.id, "", output / "control")
        passed = gold["reward"] == 1 and control["reward"] == 0
        result = {"passed": passed, "task_id": task["task_id"], "offline": offline,
                  "gold": gold, "control": control, "model_api_calls": 0, "finished_at": now()}
        save(root / "preflight/readiness.json", result)
        if not passed:
            raise RuntimeError("Official gold/control preflight failed")
        return result
    finally:
        client.close()


def run_batch(root, tasks, config, workers):
    if not json.loads((root / "preflight/readiness.json").read_text())["passed"]:
        raise ValueError("Offline and official scorer validation must pass before model generation")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Inject DEEPSEEK_API_KEY in the host process environment")
    lock = threading.Lock()
    states = {task["task_id"]: {"phase": "queued"} for task in tasks}

    def update(task, phase, **values):
        with lock:
            states[task["task_id"]].update(phase=phase, updated_at=now(), **values)
            save(root / "monitor-status.json", {"workers": workers, "task_count": len(tasks),
                                                "updated_at": now(), "tasks": states})
        print(json.dumps({"task_id": task["task_id"], "phase": phase, **values}), flush=True)

    def one(task):
        output = root / "outputs" / task["task_id"]
        output.mkdir(parents=True, exist_ok=True)
        if (output / "result.json").exists():
            result = json.loads((output / "result.json").read_text())
            update(task, result["status"], reused_existing_outcome=True)
            return result
        client, added = docker.from_env(timeout=1800), False
        try:
            update(task, "image_preparation")
            image, added = get_image(client, task, output)
            update(task, "solving", started_at=now())
            model_patch, info = generate(task, root, output, config, image, "generate")
            update(task, "official_evaluation", exit_status=info.get("exit_status"))
            result = grade(root, task, image.id, model_patch, output / "evaluation")
            result.update(exit_status=info.get("exit_status"), patch_sha256=hashlib.sha256(model_patch.encode()).hexdigest(),
                          finished_at=now(), native_agent_completed=info.get("exit_status") == "Submitted")
        except Exception as error:  # noqa: BLE001 - retain per-task infrastructure/stage outcomes
            result = {"status": "infrastructure_or_stage_error", "task_id": task["task_id"],
                      "error_type": type(error).__name__, "error": str(error), "reward": None,
                      "finished_at": now()}
        finally:
            if added:
                try:
                    client.images.remove(task["image"], noprune=False)
                except docker.errors.APIError:
                    pass  # Preserve shared layers or another live container.
            client.close()
        save(output / "result.json", result)
        update(task, result["status"], reward=result.get("reward"))
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [future.result() for future in as_completed([pool.submit(one, task) for task in tasks])]
    save(root / "result.json", {"task_count": len(tasks), "results": results, "finished_at": now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--task-id", action="append", help="Run only named cases from the frozen FT30")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.workers != 3:
        raise ValueError("This frozen FT30 experiment was approved with three pipeline slots")
    tasks, config = load_tasks(root), prepare_config(root)
    if args.task_id:
        selected = set(args.task_id)
        if selected - {task["task_id"] for task in tasks}:
            raise ValueError("Requested task is outside the frozen FT30")
        tasks = [task for task in tasks if task["task_id"] in selected]
    if args.mode == "prepare":
        save(root / "preflight/plan.json", {"task_count": len(tasks), "workers": args.workers,
                                           "model_api_calls": 0, "agent": config["agent"],
                                           "same_ft_task_checksums": True})
    elif args.mode == "preflight":
        preflight(root, tasks[0], config)
    else:
        run_batch(root, tasks, config, args.workers)


if __name__ == "__main__":
    main()
