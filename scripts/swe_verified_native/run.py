"""Experiment bookkeeping around unmodified native mini-SWE-agent components."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import io
import json
import os
import shlex
import shutil
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import minisweagent
import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import get_environment
from minisweagent.models import get_model

import docker


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, ensure_ascii=False, indent=2)
    credential = os.environ.get("DEEPSEEK_API_KEY")
    if credential:
        text = text.replace(credential, "[REDACTED]")
    temporary.write_text(text + "\n")
    temporary.replace(path)


def image_name(task):
    return "docker.io/swebench/sweb.eval.x86_64." + task["instance_id"].replace("__", "_1776_").lower() + ":latest"


def load_tasks(root):
    manifest = json.loads((root / "manifests/verified_random100_seed20261026.json").read_text())
    tasks = [json.loads(s) for s in (root / "manifests/public_tasks.jsonl").read_text().splitlines()]
    if len(tasks) != 100 or [t["instance_id"] for t in tasks] != manifest["task_ids"]:
        raise ValueError("The frozen 100-case manifest and public tasks differ")
    expected = {"instance_id", "repo", "base_commit", "version", "problem_statement"}
    if any(set(task) != expected for task in tasks):
        raise ValueError("Generation must receive only the five public task fields")
    return tasks


def prepare_config(root):
    upstream = Path(minisweagent.__file__).parent / "config/benchmarks/swebench.yaml"
    stock = yaml.safe_load(upstream.read_text())
    config = copy.deepcopy(stock)
    config["agent"].update(step_limit=250, cost_limit=0, wall_time_limit_seconds=10800)
    config["environment"].update(
        run_args=["--rm", "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                  "--cpus=2", "--memory=6g", "--pids-limit=512"],
        # The lifetime covers preparation plus the approved three-hour agent window.
        container_timeout="4h", forward_env=[],
    )
    config["model"].update(model_name="deepseek/deepseek-flash", cost_tracking="ignore_errors",
                            model_class="native_compat.DeepSeekNativeModel")
    config["model"]["model_kwargs"].update(
        api_base="https://api.deepseek.com/v1", reasoning_effort="high", max_tokens=65536,
        extra_body={"thinking": {"type": "enabled"}},
    )
    destination = root / "configs/native-deepseek-v4.1-flash.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and yaml.safe_load(destination.read_text()) != config:
        raise ValueError("Existing native configuration differs; review changes before replacing it")
    destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    (destination.parent / "upstream-swebench.yaml").write_bytes(upstream.read_bytes())
    return config, stock


def upload(container, source, destination):
    stream = io.BytesIO()
    data = source.read_bytes()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(Path(destination).name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    if not container.put_archive(str(Path(destination).parent), stream.getvalue()):
        raise RuntimeError("Could not upload baseline isolation helper")


def prepare_repository(root, environment, container, task):
    upload(container, root / "scripts/sanitize_repository.py", "/tmp/native_sanitize_repository.py")
    result = environment.execute({"command": "python /tmp/native_sanitize_repository.py --revision "
                                  + shlex.quote(task["base_commit"])}, timeout=300)
    if result["returncode"]:
        raise RuntimeError("Baseline isolation failed: " + result["output"][-1800:])
    audit = json.loads(result["output"].splitlines()[-1])
    other = environment.execute({"command": "find /testbed /root /opt -name .git -type d -prune 2>/dev/null || true"})
    directories = [p for p in other["output"].splitlines() if p != "/testbed/.git"]
    for directory in directories:
        result = environment.execute({"command": "rm -rf -- " + shlex.quote(directory)})
        if result["returncode"]:
            raise RuntimeError("Could not remove another repository history")
    check = environment.execute({"command": "python - <<'PY'\nimport socket\ns=socket.socket();s.settimeout(2)\ntry:s.connect(('1.1.1.1',443));print('NETWORK_OPEN')\nexcept OSError:print('NETWORK_BLOCKED')\nfinally:s.close()\nPY"})
    container.reload()
    host = container.attrs["HostConfig"]
    if "NETWORK_BLOCKED" not in check["output"] or host["NetworkMode"] != "none":
        raise RuntimeError("The native agent container has network access")
    if host.get("Binds") or host.get("Privileged") or environment.config.forward_env:
        raise RuntimeError("The native container has unexpected host access")
    audit.update(network="none", other_git_directories_removed=len(directories),
                 container_cpus=host["NanoCpus"], container_memory=host["Memory"],
                 host_mounts=host.get("Binds") or [], forwarded_host_environment=[])
    return audit


def make_agent(config, environment, output):
    model = get_model(config=config["model"])
    options = dict(config["agent"], output_path=output / "native.traj.json")
    return DefaultAgent(model, environment, **options)


def offline_check(config, environment, output, task):
    import litellm

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        assert [t["function"]["name"] for t in kwargs["tools"]] == ["bash"]
        if len(calls) == 2:
            assistant = [m for m in kwargs["messages"] if m["role"] == "assistant"][-1]
            assert assistant["reasoning_content"] == " "
        command = "printf 'NATIVE_MINI_OFFLINE_OK\\n'" if len(calls) == 1 else (
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && printf 'OFFLINE_PATCH_ONLY\\n'"
        )
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Offline protocol probe.", "reasoning_content": "",
                "tool_calls": [{"id": "offline-" + str(len(calls)), "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
            },
        }], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    agent = make_agent(config, environment, output)
    with patch("litellm.completion", completion):
        info = agent.run(task["problem_statement"])
    assert info["exit_status"] == "Submitted" and info["submission"] == "OFFLINE_PATCH_ONLY\n"
    first = next(m for m in agent.messages if m["role"] == "assistant")
    assert first["reasoning_content"] == ""
    return {"status": "offline_passed", "model_api_calls": 0, "simulated_model_responses": len(calls),
            "native_agent": type(agent).__module__ + "." + type(agent).__name__,
            "native_environment": type(environment).__module__ + "." + type(environment).__name__,
            "outbound_compatibility_verified": True, "raw_reasoning_preserved": True,
            "note": "Protocol check only; not a generated benchmark patch or score."}


def run_one(task, root, output_root, config, mode, *, retain_image=False):
    output = output_root / task["instance_id"]
    output.mkdir(parents=True, exist_ok=True)
    if mode == "generate" and (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    client = docker.from_env()
    environment = None
    added = False
    image = image_name(task)
    try:
        save(output / "progress.json", {"phase": "image_preparation", "model_api_calls": 0})
        try:
            reference = client.images.get(image)
        except docker.errors.ImageNotFound:
            # Pull before native docker run; retain the upstream startup timeout.
            if shutil.disk_usage(root).free < 15 * 1024**3:
                raise RuntimeError("Less than 15 GiB free; stop before pulling another image")
            reference = client.images.pull(image)
            added = True
        save(output / "image.json", {"name": image, "id": reference.id, "pulled_by_runner": added,
                                      "repo_digests": reference.attrs.get("RepoDigests", [])})
        options = dict(config["environment"], image=image)
        environment = get_environment(options)
        container = client.containers.get(environment.container_id)
        isolation = prepare_repository(root, environment, container, task)
        save(output / "isolation.json", isolation)
        if mode == "preflight":
            probe = environment.execute({"command": "python -c 'import sys; print(sys.executable); print(sys.version)'"})
            if probe["returncode"]:
                raise RuntimeError(probe["output"])
            result = {"instance_id": task["instance_id"], "status": "prepared", "model_api_calls": 0,
                      "python": probe["output"], "isolation": isolation}
        elif mode == "offline-check":
            result = {"instance_id": task["instance_id"], **offline_check(config, environment, output, task)}
        else:
            save(output / "progress.json", {"phase": "solving", "framework": "native-mini-swe-agent"})
            agent = make_agent(config, environment, output)
            info = agent.run(task["problem_statement"])
            prediction = {"instance_id": task["instance_id"], "model_name_or_path": "native-mini-deepseek-v4.1-flash",
                          "model_patch": info.get("submission") or ""}
            save(output / "prediction.json", prediction)
            result = {"instance_id": task["instance_id"], "status": "generated",
                      "exit_status": info.get("exit_status"), "empty_patch": not prediction["model_patch"].strip()}
        save(output / "result.json", result)
        return result
    except Exception as error:  # noqa: BLE001 - keep infrastructure failures separate from scores
        result = {"instance_id": task["instance_id"], "status": "infrastructure_or_stage_error",
                  "error_type": type(error).__name__, "error": str(error)}
        save(output / "result.json", result)
        return json.loads((output / "result.json").read_text())
    finally:
        if environment is not None and environment.container_id:
            try:
                client.containers.get(environment.container_id).remove(force=True)
            except docker.errors.NotFound:
                pass  # Native --rm may already have removed a stopped container.
            finally:
                environment.container_id = None
        if added and mode == "generate" and not retain_image:
            try:
                client.images.remove(image, noprune=False)
            except docker.errors.APIError:
                pass
        client.close()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def solve_evaluate(task, root, output_root, config, reference, run_id, on_update=lambda: None):
    """One slot owns solving, isolated official grading, then image cleanup."""
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    output = output_root / task["instance_id"]
    evaluation_path = output / "evaluation.json"
    client = None
    try:
        generation = run_one(task, root, output_root, config, "generate", retain_image=True)
        # run_one's finally has removed the agent container before returning.
        if generation["status"] != "generated":
            evaluation = {"instance_id": task["instance_id"], "status": "not_evaluated",
                          "reason": "generation_error", "finished_at": now()}
            save(evaluation_path, evaluation)
            return {**generation, "evaluation_status": evaluation["status"]}
        prediction_path = output / "prediction.json"
        prediction_bytes = prediction_path.read_bytes()
        prediction = json.loads(prediction_bytes)
        if prediction["instance_id"] != task["instance_id"]:
            raise ValueError("P0 prediction belongs to another task")
        digest = hashlib.sha256(prediction_bytes).hexdigest()
        previous = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
        if previous.get("prediction_sha256") not in (None, digest):
            raise ValueError("P0 changed since its previous evaluation")
        if previous.get("status") in {"resolved", "unresolved", "empty_patch", "evaluation_error"}:
            if previous.get("run_id") != run_id:
                raise ValueError("Existing evaluation belongs to another run ID")
            return {**generation, "evaluation_status": previous["status"]}
        evaluation = {"instance_id": task["instance_id"], "status": "evaluating", "run_id": run_id,
                      "started_at": now(), "prediction_sha256": digest,
                      "patch_sha256": hashlib.sha256(prediction["model_patch"].encode()).hexdigest(),
                      "generation_cleanup_completed": True, "timeout_sec": 1800}
        save(evaluation_path, evaluation)
        save(output / "progress.json", {"phase": "official_evaluation", "updated_at": now()})
        on_update()
        if prediction["model_patch"] == "":
            evaluation["status"] = "empty_patch"
        else:
            spec = make_test_spec(reference, namespace="swebench")
            model = prediction["model_name_or_path"].replace("/", "__")
            report_path = Path.cwd() / "logs/run_evaluation" / run_id / model / task["instance_id"] / "report.json"
            evaluation["official_report_path"] = str(report_path)
            save(evaluation_path, evaluation)
            if report_path.exists() and previous.get("prediction_sha256") != digest:
                raise ValueError("Existing official report has no matching P0 receipt")
            client = docker.from_env()
            if not report_path.exists():
                # Use the official evaluator unchanged. This slot retains its
                # generation image and cleans up only that image afterwards.
                run_instance(spec, prediction, False, False, client, run_id, timeout=1800)
            if not report_path.exists():
                raise RuntimeError("Official evaluator ended without report.json; inspect run_instance.log")
            report = json.loads(report_path.read_text())[task["instance_id"]]
            if not isinstance(report.get("resolved"), bool):
                raise ValueError("Official report has no boolean resolved field")
            evaluation["status"] = "resolved" if report["resolved"] else "unresolved"
            evaluation["test_counts"] = {
                name: {key: len(result.get(key, [])) for key in ["success", "failure"]}
                for name, result in report.get("tests_status", {}).items()
            }
            if prediction_path.read_bytes() != prediction_bytes:
                raise ValueError("P0 changed while the official evaluator was running")
        evaluation["finished_at"] = now()
        save(evaluation_path, evaluation)
        save(output / "progress.json", {"phase": "finished", "evaluation_status": evaluation["status"], "updated_at": now()})
        return {**generation, "evaluation_status": evaluation["status"]}
    except Exception as error:  # noqa: BLE001 - preserve scoring errors separately from P0 generation
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {"instance_id": task["instance_id"]}
        evaluation.update(status="evaluation_error", error_type=type(error).__name__, error=str(error), finished_at=now())
        save(evaluation_path, evaluation)
        save(output / "progress.json", {"phase": "finished", "evaluation_status": "evaluation_error", "updated_at": now()})
        return {"instance_id": task["instance_id"], "status": "workflow_error", "evaluation_status": "evaluation_error"}
    finally:
        if client is not None:
            client.close()
        image_path = output / "image.json"
        if image_path.exists():
            image = json.loads(image_path.read_text())
            if image.get("pulled_by_runner"):
                cleanup = docker.from_env()
                try:
                    cleanup.images.remove(image["name"], noprune=False)
                except docker.errors.APIError:
                    pass
                finally:
                    cleanup.close()
        on_update()


def write_pipeline_status(output, tasks, workers):
    from collections import Counter

    generation, evaluation, active = Counter(), Counter(), {}
    for task in tasks:
        directory = output / task["instance_id"]
        for name, counts in [("result.json", generation), ("evaluation.json", evaluation)]:
            path = directory / name
            if path.exists():
                counts[json.loads(path.read_text())["status"]] += 1
        path = directory / "progress.json"
        if path.exists():
            phase = json.loads(path.read_text())["phase"]
            if phase != "finished" and not (directory / "evaluation.json").exists():
                active[task["instance_id"]] = phase
            elif phase == "official_evaluation":
                active[task["instance_id"]] = phase
    save(output.parent / "monitor-status.json", {
        "updated_at": now(), "task_count": len(tasks), "pipeline_workers": workers,
        "scheduling": "solve_then_official_evaluate_per_slot", "generation_counts": dict(generation),
        "evaluation_counts": dict(evaluation), "active": active,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "offline-check", "generate", "solve-evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-id", help="Official grading run ID; required for solve-evaluate")
    args = parser.parse_args()
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    tasks = load_tasks(args.root)
    config, stock = prepare_config(args.root)
    if args.task_id:
        if set(args.task_id) - {t["instance_id"] for t in tasks}:
            parser.error("Task outside the frozen random100")
        tasks = [t for t in tasks if t["instance_id"] in args.task_id]
    if args.mode == "prepare":
        assert config["agent"]["system_template"] == stock["agent"]["system_template"]
        assert config["agent"]["instance_template"] == stock["agent"]["instance_template"]
        for field in ["observation_template", "format_error_template"]:
            assert config["model"][field] == stock["model"][field]
        save(args.root / "preflight/native-config-check.json", {
            "task_count": len(tasks), "stock_prompts_and_observation_templates_unchanged": True,
            "native_tool_names": ["bash"], "model_api_calls": 0,
            "upstream_config_sha256": hashlib.sha256((args.root / "configs/upstream-swebench.yaml").read_bytes()).hexdigest(),
        })
        print(json.dumps({"status": "configured", "task_count": len(tasks), "model_api_calls": 0}))
        return
    if args.mode in {"preflight", "offline-check"} and not args.task_id:
        parser.error("Checks require explicit --task-id; they never start all100")
    generation_mode = args.mode in {"generate", "solve-evaluate"}
    if generation_mode and (not args.output or not args.workers or not os.environ.get("DEEPSEEK_API_KEY")):
        parser.error("Generation requires an explicit --output, --workers and host DEEPSEEK_API_KEY")
    if args.mode == "solve-evaluate" and not args.run_id:
        parser.error("solve-evaluate requires an explicit --run-id")
    output = args.output or args.root / "preflight" / args.mode
    workers = args.workers if generation_mode else 1
    lock = threading.Lock()

    def on_update():
        with lock:
            write_pipeline_status(output, tasks, workers)

    references = {}
    if args.mode == "solve-evaluate":
        # Set the host working directory once, before any worker starts.
        # Official logs are host-only and never uploaded to the agent containers.
        args.root = args.root.resolve()
        output = output.resolve()
        verifier = output.parent / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        references = {row["instance_id"]: row for row in json.loads((args.root / "grader/reference_tasks.json").read_text())}
        if not {task["instance_id"] for task in tasks}.issubset(references):
            parser.error("Official reference dataset does not cover all selected tasks")
        os.chdir(verifier)
        save(output.parent / "evaluation-config.json", {
            "run_id": args.run_id, "pipeline_workers": workers,
            "scheduling": "solve_then_official_evaluate_per_slot", "official_timeout_sec": 1800,
            "official_api": "swebench.harness.run_evaluation.run_instance",
            "separate_generation_and_scoring_containers": True, "model_api_feedback": False,
        })
        on_update()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if args.mode == "solve-evaluate":
            futures = [pool.submit(solve_evaluate, task, args.root, output, config,
                                   references[task["instance_id"]], args.run_id, on_update) for task in tasks]
        else:
            futures = [pool.submit(run_one, task, args.root, output, config, args.mode) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result, ensure_ascii=False), flush=True)
    if generation_mode:
        predictions = [json.loads(p.read_text()) for p in sorted(output.glob("*/prediction.json"))]
        (output / "predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions))
    if args.mode == "solve-evaluate":
        from swebench.harness.reporting import make_run_report

        selected = {task["instance_id"] for task in tasks}
        prediction_map = {row["instance_id"]: row for row in predictions if row["instance_id"] in selected}
        if prediction_map:
            client = docker.from_env()
            try:
                make_run_report(prediction_map, [references[task["instance_id"]] for task in tasks],
                                args.run_id, client, namespace="swebench")
            finally:
                client.close()
        on_update()


if __name__ == "__main__":
    main()
