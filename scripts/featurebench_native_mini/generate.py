"""Stock Mini Agent/Bash; official FeatureBench initialization happens before it."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path

import docker
import minisweagent
import yaml
from minisweagent.agents.default import AgentConfig
from minisweagent.environments import get_environment

import native_core as native
from common import offline_agent_info, save
from iers_data import install_leap_seconds
from featurebench.infer.container import ContainerManager
from featurebench.infer.models import TaskInstance
from featurebench.infer.runtime import RuntimeHandler


def configuration(root):
    upstream = yaml.safe_load((Path(minisweagent.__file__).parent / "config/mini.yaml").read_text())
    config = copy.deepcopy(upstream)
    config["agent"] = {key: value for key, value in config["agent"].items() if key in AgentConfig.model_fields}
    config["agent"].update(step_limit=0, cost_limit=0, wall_time_limit_seconds=10800)
    config["model"].update(model_name="deepseek/deepseek-flash", cost_tracking="ignore_errors",
                           model_class="native_compat.DeepSeekNativeModel")
    config["model"]["model_kwargs"].update(
        api_base="https://api.deepseek.com/v1", reasoning_effort="high", max_tokens=65536,
        extra_body={"thinking": {"type": "enabled"}},
    )
    config["environment"].update(
        environment_class="docker", cwd="/testbed", forward_env=[], container_timeout="4h",
        run_args=["--rm", "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                  "--pids-limit=512", "--cpus=2", "--memory=6g", "--entrypoint="],
        interpreter=["bash", "-lc", "source /opt/miniconda3/etc/profile.d/conda.sh && "
                     "conda activate testbed && "
                     "export PYTHONPATH=/testbed:/testbed/src${PYTHONPATH:+:$PYTHONPATH} && eval \"$1\"",
                     "native-featurebench"],
    )
    path = root / "configs/native-deepseek-v4.1-flash.yaml"
    path.parent.mkdir(exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text()) != config:
        raise ValueError("Native experiment configuration changed")
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    for name in ["system_template", "instance_template"]:
        assert config["agent"][name] == upstream["agent"][name]
    return config


def prepare(root, environment, container, task, output):
    class Manager(ContainerManager):
        def exec_command(self, container, command, timeout=None, workdir=None, log_file=None):
            result = environment.execute({"command": command}, cwd=workdir or "/testbed", timeout=timeout or 600)
            if log_file:
                with Path(log_file).open("a") as log:
                    log.write(f"$ {command}\n{result['output']}\n[Exit code: {result['returncode']}]\n")
            return result["returncode"], result["output"]

    manager = Manager(logger=logging.getLogger("native.featurebench.preparation"))
    try:
        handler = RuntimeHandler(manager)
        instance = TaskInstance.from_dict(task)
        log = output / "official-preparation.log"
        if task["level"] == 1:
            prepared = handler._initialize_level1(container, instance, log, white_box=False)
        else:
            prepared = handler._initialize_level2(container, instance, log)
        if not prepared or not handler.clear_package_caches(container, log):
            raise RuntimeError("Official task initialization failed")
    finally:
        manager.client.close()
    native.upload(container, root / "scripts/sanitize_repository.py", "/tmp/native_sanitize.py")
    audit = json.loads(native.checked(environment, "python /tmp/native_sanitize.py --revision HEAD").splitlines()[-1])
    paths = native.checked(environment, "find /testbed /root /opt -name .git -print -prune 2>/dev/null || true")
    for path in paths.splitlines():
        if path != "/testbed/.git":
            native.checked(environment, "rm -rf -- " + __import__("shlex").quote(path))
    forbidden = ["/root/my_repo", "/tmp/mask.patch", "/tmp/mask_patch.diff", "/tmp/test_patch.diff"]
    forbidden += ["/testbed/" + p.removeprefix("/testbed/") for p in task["FAIL_TO_PASS"]]
    script = ("import os,socket,json\n" + f"paths={forbidden!r}\n"
              "s=socket.socket();s.settimeout(2)\n"
              "try:s.connect(('1.1.1.1',443));blocked=False\n"
              "except OSError:blocked=True\n"
              "finally:s.close()\n"
              "print(json.dumps({'network_blocked':blocked,'forbidden_remaining':[p for p in paths if os.path.exists(p)]}))")
    probe = json.loads(native.checked(environment, "python -c " + __import__("shlex").quote(script)).splitlines()[-1])
    container.reload()
    host = container.attrs["HostConfig"]
    if not probe["network_blocked"] or probe["forbidden_remaining"] or host.get("Binds") or host["NetworkMode"] != "none":
        raise RuntimeError("Task isolation audit failed")
    audit.update(**probe, native_agent="DefaultAgent", tools=["bash"],
                 baseline="official post-initialization synthetic commit", credentials_in_container=False,
                 container_cpus=2, container_memory_mb=6144, ledger_included=False)
    save(output / "isolation.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    task = json.loads(args.task.read_text())
    config = configuration(root)
    options = copy.deepcopy(config["environment"])
    options["image"] = args.image_id
    runtime = TaskInstance.from_dict(task).get_docker_runtime_config()
    options["env"].update(runtime["env_vars"])
    for assignment in runtime["env_exports"]:
        key, value = assignment.removeprefix("export ").split("=", 1)
        options["env"][key] = value
    if runtime["shm_size"]:
        options["run_args"].append("--shm-size=" + runtime["shm_size"])
    environment = get_environment(options)
    client = docker.from_env()
    try:
        container = client.containers.get(environment.container_id)
        install_leap_seconds(container, root / "configs/iers-leap-seconds.json",
                             output / "iers-data.json")
        audit = prepare(root, environment, container, task, output)
        public = {"instruction": task["problem_statement"]}
        if args.offline:
            diagnostics = native.offline_probe(config, environment, output, public)
            info = offline_agent_info(output, diagnostics)
        else:
            info = native.make_agent(config, environment, output).run(task["problem_statement"])
        patch = native.checked(environment, native.snapshot_command(audit["synthetic_commit"]))
        (output / "model.patch").write_text(patch)
        save(output / "generation.json", {"instance_id": task["instance_id"], "agent_info": info,
             "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "patch_bytes": len(patch.encode()),
             "model_api_calls": 0 if args.offline else None, "ledger_included": False})
    finally:
        if environment.container_id:
            try:
                client.containers.get(environment.container_id).remove(force=True)
            except docker.errors.NotFound:
                pass
            environment.container_id = None
        client.close()


if __name__ == "__main__":
    main()
