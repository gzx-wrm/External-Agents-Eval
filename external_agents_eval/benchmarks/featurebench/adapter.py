"""Prepare official Level 1 tasks for the unchanged Mini V6.3 workflow."""
from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path

from external_agents_eval.benchmarks.swebench.isolated_environment import IsolatedEnvironment


class FeatureBenchEnvironment(IsolatedEnvironment):
    """Activate the image's testbed environment and import the active worktree."""

    def exec_sync(self, command, *, cwd=None, timeout_sec=None, **kwargs):
        directory = cwd or "/testbed"
        paths = shlex.quote(directory + ":" + directory + "/src")
        wrapped = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed || exit $?\n"
            f"export PYTHONPATH={paths}${{PYTHONPATH:+:$PYTHONPATH}}\n" + command
        )
        return super().exec_sync(wrapped, cwd=directory, timeout_sec=timeout_sec, **kwargs)


def task_container(client, image, task, *, name):
    from featurebench.infer.models import TaskInstance

    config = TaskInstance.from_dict(task).get_docker_runtime_config()
    if task.get("level") != 1 or config["need_gpu"]:
        raise ValueError("This initial adapter supports CPU-only FeatureBench Level 1 tasks")
    environment = {"PAGER": "cat", "GIT_TERMINAL_PROMPT": "0", **config["env_vars"]}
    # Preserve task-specific exports without enabling network or host mounts.
    for assignment in config["env_exports"]:
        key, value = assignment.removeprefix("export ").split("=", 1)
        environment[key] = value
    return client.containers.create(
        image, command=["sleep", "infinity"], entrypoint=[], name=name,
        working_dir="/testbed", network_mode="none", cap_drop=["ALL"],
        security_opt=["no-new-privileges:true"], privileged=False,
        nano_cpus=2_000_000_000, mem_limit="6g", pids_limit=512,
        shm_size=config["shm_size"] or "64m", environment=environment,
        labels={"external_eval": "v63-featurebench", "task": task["instance_id"]},
    )


async def prepare_task(environment, task, output):
    """Reuse the pinned official masking/cleanup, then sanitize its resulting HEAD."""
    from featurebench.infer.container import ContainerManager
    from featurebench.infer.models import TaskInstance
    from featurebench.infer.runtime import RuntimeHandler

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    class RuntimeManager(ContainerManager):
        def exec_command(self, container, command, timeout=None, workdir=None, log_file=None):
            result = environment.exec_sync(command, cwd=workdir, timeout_sec=timeout or 600)
            text = result.stdout + result.stderr
            if log_file:
                with Path(log_file).open("a") as log:
                    log.write(f"$ {command}\n{text}\n[Exit code: {result.return_code}]\n")
            return result.return_code, text

    logger = logging.getLogger("featurebench.preparation")
    manager = RuntimeManager(logger=logger)
    try:
        handler = RuntimeHandler(manager, logger)
        log = output / "official-preparation.log"
        instance = TaskInstance.from_dict(task)
        # The full official initialize_runtime also installs terminal recording
        # tools. Those are unnecessary for a host-side agent in an offline image.
        if not handler._initialize_level1(environment.container, instance, log, white_box=False):
            raise RuntimeError("Official Level 1 task preparation failed; inspect preparation log")
        if not handler.clear_package_caches(environment.container, log):
            raise RuntimeError("Official package-cache cleanup failed")
    finally:
        manager.client.close()

    source = Path(__file__).parents[1] / "swebench/sanitize_repository.py"
    await environment.upload_file(source, "/tmp/featurebench_sanitize.py")
    result = await environment.exec(
        "python /tmp/featurebench_sanitize.py --revision HEAD", timeout_sec=600,
    )
    if result.return_code:
        raise RuntimeError("Post-mask Git sanitization failed: " + result.stderr[-2000:])
    audit = json.loads(result.stdout.splitlines()[-1])
    scan = await environment.exec(
        "find /testbed /root /opt -name .git -type d -prune 2>/dev/null || true", timeout_sec=120,
    )
    other = [path for path in scan.stdout.splitlines() if path != "/testbed/.git"]
    for path in other:
        removed = await environment.exec("rm -rf -- " + shlex.quote(path), timeout_sec=120)
        if removed.return_code:
            raise RuntimeError("Could not remove auxiliary Git history")
    audit["other_git_directories_removed"] = len(other)
    forbidden = ["/root/my_repo", "/tmp/mask.patch", "/tmp/mask_patch.diff", "/tmp/test_patch.diff"]
    forbidden += ["/testbed/" + path.removeprefix("/testbed/") for path in task["FAIL_TO_PASS"]]
    probe = (
        "import json,os,socket\n"
        f"paths={forbidden!r}\n"
        "left=[p for p in paths if os.path.exists(p)]\n"
        "s=socket.socket();s.settimeout(2)\n"
        "try:s.connect(('1.1.1.1',443));blocked=False\n"
        "except OSError:blocked=True\n"
        "finally:s.close()\n"
        "print(json.dumps({'forbidden_paths_remaining':left,'network_blocked':blocked}))\n"
    )
    result = await environment.exec("python -c " + shlex.quote(probe))
    if result.return_code:
        raise RuntimeError("Task isolation probe failed")
    audit.update(json.loads(result.stdout.splitlines()[-1]))
    environment.container.reload()
    host = environment.container.attrs["HostConfig"]
    if (host["NetworkMode"] != "none" or host.get("Privileged") or host.get("Binds")
            or audit["forbidden_paths_remaining"] or not audit["network_blocked"]):
        raise RuntimeError("FeatureBench task isolation audit failed")
    audit.update(
        network="none", baseline="official post-mask, post-cleanup synthetic commit",
        official_white_box=False, credentials_in_container=False,
        preparation="official _initialize_level1 + clear_package_caches; terminal recorder install omitted",
    )
    (output / "isolation.json").write_text(json.dumps(audit, indent=2))
    return audit


async def check_worktree_imports(environment, audit, output):
    """Verify active-tree imports and real relative/absolute A read evidence."""
    from external_agents_eval.agents.mini_v62 import MiniWorkspaceManager
    from external_agents_eval.agents.pi_file_tools import PI_EVIDENCE_PROTOCOL, prepare_tools
    from external_agents_eval.benchmarks.featurebench.agents import FeatureBenchPiFileEnvironment
    from external_agents_eval.workflows.solve_review_repair.ft_evidence import observed_evidence

    workspace = MiniWorkspaceManager(
        environment=environment, repository="/testbed", transfer_dir=Path(output) / "transfers",
    )
    results = {}
    try:
        paths = await workspace.prepare(base_commit=audit["synthetic_commit"])
        for role in ["solver", "spec_review", "patch_review", "repair", "summary"]:
            cwd = getattr(paths, role)
            script = (
                "import packaging,json,os,sys\n"
                "p=os.path.realpath(packaging.__file__)\n"
                "assert p.startswith(os.getcwd()+'/'),p\n"
                "print(json.dumps({'module':p,'python':sys.executable}))\n"
            )
            result = await environment.exec("python -c " + shlex.quote(script), cwd=cwd)
            if result.return_code:
                raise RuntimeError(f"{role} imports an incorrect tree: {result.stdout}{result.stderr}")
            results[role] = json.loads(result.stdout.splitlines()[-1])
        await prepare_tools(environment)
        tools = FeatureBenchPiFileEnvironment(environment, paths.spec_review)
        source = "src/packaging/__init__.py"
        for index, path in enumerate([source, "/testbed/" + source]):
            result = tools.execute({"name": "read", "arguments": {"path": path, "limit": 20},
                                    "tool_call_id": f"preflight-read-{index}"})
            if result["returncode"]:
                raise RuntimeError("Reference Pi read failed: " + result["output"])
        steps = []
        for index, (name, arguments, result) in enumerate(tools.events):
            call_id = f"preflight-read-{index}"
            steps.append({
                "step_id": index + 1,
                "tool_calls": [{"tool_call_id": call_id, "function_name": name, "arguments": arguments}],
                "observation": {"results": [{"source_call_id": call_id, "content": result["output"],
                                              "exit_code": result["returncode"]}]},
            })
        evidence = await observed_evidence(
            {"evidence_protocol": PI_EVIDENCE_PROTOCOL, "steps": steps}, task_id="preflight",
            base_root=paths.spec_review, base_commit=audit["synthetic_commit"], workspace=workspace,
        )
        if evidence["skipped_reads"] or len(evidence["read_provenance"]) != 2 or not evidence["evidence"]:
            raise RuntimeError("A path alias or frozen evidence collection lost a successful read")
        (Path(output) / "pi-evidence-probe.json").write_text(json.dumps(evidence, indent=2))
    finally:
        await workspace.cleanup()
    (Path(output) / "worktree-imports.json").write_text(json.dumps(results, indent=2))
    return results
