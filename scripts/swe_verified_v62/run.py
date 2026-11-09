"""Host-side preparation/generation for isolated mini-SWE-agent V6.2 on Verified."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import docker
from external_agents_eval.agents.mini_v62 import MiniV62Agent
from external_agents_eval.benchmarks.swebench import sanitize_repository
from external_agents_eval.benchmarks.swebench.isolated_environment import (
    IsolatedEnvironment,
    task_container,
)
from external_agents_eval.benchmarks.swebench.manifest import atomic_write_json


def image_name(instance_id):
    return "docker.io/swebench/sweb.eval.x86_64." + instance_id.lower().replace("__", "_1776_") + ":latest"


async def prepare_repository(environment, entry):
    await environment.upload_file(Path(sanitize_repository.__file__), "/tmp/sanitize_repository.py")
    result = await environment.exec(
        "python /tmp/sanitize_repository.py --revision " + entry["base_commit"],
        cwd="/testbed", timeout_sec=300,
    )
    if result.return_code:
        raise RuntimeError("Repository sanitization failed: " + result.stderr[-2000:])
    audit = json.loads(result.stdout.splitlines()[-1])
    # Remove separate repository histories baked into the image, if present.
    scan = await environment.exec(
        "find /testbed /root /opt -name .git -type d -prune 2>/dev/null || true", timeout_sec=60,
    )
    other = [p for p in scan.stdout.splitlines() if p != "/testbed/.git"]
    if other:
        import shlex
        for path in other:
            await environment.exec("rm -rf -- " + shlex.quote(path))
    audit["other_git_directories_removed"] = len(other)
    result = await environment.exec(
        "python - <<'PY'\n"
        "import socket\n"
        "s=socket.socket();s.settimeout(2)\n"
        "try:s.connect(('1.1.1.1',443));print('NETWORK_OPEN')\n"
        "except OSError:print('NETWORK_BLOCKED')\n"
        "finally:s.close()\nPY",
    )
    if "NETWORK_BLOCKED" not in result.stdout:
        raise RuntimeError("Agent container unexpectedly has network access")
    environment.container.reload()
    host = environment.container.attrs["HostConfig"]
    if host["NetworkMode"] != "none" or host.get("Privileged") or host.get("Binds"):
        raise RuntimeError("Task container isolation configuration changed")
    audit["network"] = "none"
    return audit


def run_one(entry, args, *, preflight=False):
    client = docker.from_env()
    container = None
    image_added = False
    output = args.root / ("preflight" if preflight else "outputs") / entry["instance_id"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        if shutil.disk_usage(args.root).free < 15 * 1024**3:
            raise RuntimeError("Less than 15 GiB free; stop before pulling another image")
        image = image_name(entry["instance_id"])
        try:
            reference = client.images.get(image)
        except docker.errors.ImageNotFound:
            reference = client.images.pull(image)
            image_added = True
        atomic_write_json(output / "image.json", {"name": image, "id": reference.id,
                                                  "repo_digests": reference.attrs.get("RepoDigests", [])})
        container = task_container(client, reference.id, name="sv62-" + uuid.uuid4().hex[:12])
        container.start()
        environment = IsolatedEnvironment(container, instance_id=entry["instance_id"])
        audit = asyncio.run(prepare_repository(environment, entry))
        atomic_write_json(output / "isolation.json", audit)
        if preflight:
            check = environment.exec_sync("python -c 'import sys; print(sys.version)'", cwd="/testbed")
            atomic_write_json(output / "result.json", {"status": "prepared", "python": check.stdout,
                                                        "model_calls": 0, "isolation": audit})
            return {"instance_id": entry["instance_id"], "status": "prepared"}
        if (output / "prediction.json").exists():
            return {"instance_id": entry["instance_id"], "status": "already_generated"}
        workflow = getattr(args, "workflow", "v6.2")
        agent_type = MiniV62Agent
        if workflow == "v6.3":
            from external_agents_eval.agents.mini_v63 import MiniV63Agent

            agent_type = MiniV63Agent
        agent = agent_type(
            logs_dir=output / "agent" / "logs", model_name="deepseek-flash",
            stage_model_config=args.project / "configs/v6.1-deepseek-v4.1-flash.json",
            extra_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
            fork_plan=args.fork_plan,
        )
        context = SimpleNamespace(metadata={})
        asyncio.run(agent.run(entry["problem_statement"], environment, context))
        patch_path = output / "agent/v6_review_repair/outputs/v5-p1.patch"
        if not patch_path.exists():
            raise RuntimeError(f"{workflow} ended without a final patch; inspect the stage manifest")
        prediction = {"instance_id": entry["instance_id"],
                      "model_name_or_path": workflow.replace(".", "") + "-mini-deepseek-v4.1-flash",
                      "model_patch": patch_path.read_text()}
        atomic_write_json(output / "prediction.json", prediction)
        atomic_write_json(output / "result.json", {"status": "generated", "metadata": context.metadata})
        return {"instance_id": entry["instance_id"], "status": "generated"}
    except Exception as error:  # noqa: BLE001 - keep per-task infrastructure failures explicit
        message = str(error)
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key:
            message = message.replace(key, "[REDACTED]")
        result = {"instance_id": entry["instance_id"], "status": "infrastructure_or_stage_error", "error": message}
        atomic_write_json(output / "result.json", result)
        return result
    finally:
        if container is not None:
            container.remove(force=True)
        if image_added and not preflight:
            # Only remove the image tag this invocation pulled, never global
            # caches or images that were already on the user's server.
            try:
                client.images.remove(image, noprune=False)
            except docker.errors.APIError:
                pass  # Another container may still need the shared layers.
        client.close()


def main():
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "preflight", "generate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--fork-plan", type=Path)
    parser.add_argument("--workflow", choices=["v6.2", "v6.3"], default="v6.2")
    args = parser.parse_args()
    args.project = args.project or args.root / "project"
    manifest = json.loads((args.root / "manifests/verified_random100_seed20261026.json").read_text())
    tasks = [json.loads(line) for line in (args.root / "manifests/public_tasks.jsonl").read_text().splitlines()]
    if [task["instance_id"] for task in tasks] != manifest["task_ids"]:
        raise ValueError("Public task file differs from the frozen manifest")
    if args.workers < 1 or args.workers > 3:
        raise ValueError("This evaluation runner supports one to three workers")
    if args.task_id:
        if set(args.task_id) - set(manifest["task_ids"]):
            raise ValueError("Requested task outside random100")
        tasks = [task for task in tasks if task["instance_id"] in args.task_id]
    if args.mode == "plan":
        plan = {"framework": "mini-swe-agent==2.4.6", "workflow": args.workflow, "model": "deepseek-flash",
                "workers": args.workers, "network": "none", "git_history": "synthetic baseline only",
                "model_calls": 0, "tasks": [{"instance_id": t["instance_id"], "image": image_name(t["instance_id"])} for t in tasks]}
        atomic_write_json(args.root / "preflight/plan.json", plan)
        print(json.dumps({"task_count": len(tasks), "framework": plan["framework"], "model_calls": 0}))
        return
    if args.fork_plan and len(tasks) != 1:
        parser.error("A per-task fork plan requires exactly one --task-id")
    if args.mode == "preflight" and not args.task_id:
        parser.error("preflight requires explicit --task-id; it never starts all100")
    if args.mode == "generate" and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("DEEPSEEK_API_KEY must be supplied in the host environment")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task, args, preflight=args.mode == "preflight") for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            atomic_write_json(args.root / "preflight" / (args.mode + "-status.json"), {"results": results})
            print(json.dumps(results[-1]), flush=True)
    if args.mode == "generate":
        paths = sorted((args.root / "outputs").glob("*/prediction.json"))
        (args.root / "outputs/predictions.jsonl").write_text(
            "".join(json.dumps(json.loads(path.read_text())) + "\n" for path in paths)
        )


if __name__ == "__main__":
    main()
