"""Held-out29 v6.3 on the eval30 lineage.

Per task: fresh Pi A exploration -> frozen observed evidence -> one FT ledger request
(V6.2 A path, unchanged) -> native Mini Solver with the frozen ledger appended to the
task (ft_m1, unchanged) -> official P0 score -> v6.3 Mini I/R/S/Repair (ft_m1_review,
unchanged) -> official P1 score.

prepare and preflight make no model calls; smoke-a and run are explicit. Everything is
written to a new root. The M0 held-out run and all historical runs are read-only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
import secrets
import shlex
import shutil
import sys
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.agents.config import get_agent_spec  # noqa: E402
from external_agents_eval.agents.pier.pi import _build_pi_command  # noqa: E402
from external_agents_eval.agents.pier.pi_model_config import PiStageModelConfig  # noqa: E402
from external_agents_eval.agents.pier.pi_role_runner import (  # noqa: E402
    ContainerGitWorkspaceManager,
    PiRoleRunner,
)
from external_agents_eval.agents.pier.pi_solve_review_repair import (  # noqa: E402
    issue_title_from_instruction,
)
from external_agents_eval.agents.pier.pi_v5_solve_review_repair import (  # noqa: E402
    managed_delivery_from_instruction,
)
from external_agents_eval.benchmarks.swebench.isolated_environment import (  # noqa: E402
    IsolatedEnvironment,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import (  # noqa: E402
    ArtifactStore,
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.ft_inference import (  # noqa: E402
    FTLedgerClient,
)
from external_agents_eval.workflows.solve_review_repair.models import TaskInput  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v62_config import V62Config  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v62_controller import (  # noqa: E402
    V62Controller,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The two existing entry points this experiment composes, imported unchanged.
ledger_input = load_module("v63h_ledger_input", PROJECT / "scripts/deepswe_ft_m1/ledger_input.py")
review = load_module("v63h_ft_m1_review", PROJECT / "scripts/deepswe_ft_m1_review/run.py")
read, save, load_native = review.read, review.save, review.load_native

EXPERIMENT_ID = "v6.3-heldout29-eval30-lineage"
EXCLUDED = {
    "wazero-multi-module-snapshots": (
        "M0 infrastructure error: the frozen sanitizer rejects its git submodule; "
        "the user chose to skip it on 2026-11-02"
    ),
}
STAGE_MODEL_CONFIG = PROJECT / "configs/v6.1-deepseek-v4.1-flash.json"
# PiV6/PiV62 use workflow_timeout_sec or 1800 for every role; it was never set for A.
A_ROLE_TIMEOUT_SEC = 1800
DEFAULT_EXEC_TIMEOUT_SEC = 900
PI_CLI = "/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
PI_ENV = {
    # Exactly PiV6SolveReviewRepairPierAgent.run.
    "PI_CODING_AGENT_DIR": "/external_eval/pi_config",
    "PI_OFFLINE": "1",
    "PI_SKIP_VERSION_CHECK": "1",
    "PI_TELEMETRY": "0",
    "HOME": "/external_eval/home",
    "PATH": "/opt/pi-runtime/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
EGRESS_ALIAS = "pier-egress-proxy"
EGRESS_PORT = 8080
PROXY_SCRIPT = HERE / "egress_proxy.py"
PROXY_COMMAND = 'exec "$(command -v python3 || command -v python)" /tmp/v63h_egress_proxy.py'
LABEL = {"external_eval": "v63-heldout29"}
FROZEN_SOURCES = (
    "scripts/deepswe_v63_heldout/run.py",
    "scripts/deepswe_v63_heldout/egress_proxy.py",
    "scripts/deepswe_ft_m1/ledger_input.py",
    "scripts/deepswe_ft_m1_review/run.py",
    "configs/v6.1-deepseek-v4.1-flash.json",
    "external_agents_eval/agents/config.py",
    "external_agents_eval/agents/mini_ft_review.py",
    "external_agents_eval/agents/mini_v63.py",
    "external_agents_eval/agents/mini_v62.py",
    "external_agents_eval/agents/mini_review_files.py",
    "external_agents_eval/agents/pier/pi.py",
    "external_agents_eval/agents/pier/pi_model_config.py",
    "external_agents_eval/agents/pier/pi_role_runner.py",
    "external_agents_eval/agents/pier/trajectory.py",
    "external_agents_eval/benchmarks/swebench/isolated_environment.py",
    "external_agents_eval/workflows/solve_review_repair/ft_evidence.py",
    "external_agents_eval/workflows/solve_review_repair/ft_inference.py",
    "external_agents_eval/workflows/solve_review_repair/ft_protocol.py",
    "external_agents_eval/workflows/solve_review_repair/ft/prompt.txt",
    "external_agents_eval/workflows/solve_review_repair/ft/spec_output_schema.json",
    "external_agents_eval/workflows/solve_review_repair/v5_config.py",
    "external_agents_eval/workflows/solve_review_repair/v5_contracts.py",
    "external_agents_eval/workflows/solve_review_repair/v5_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v5_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v6_config.py",
    "external_agents_eval/workflows/solve_review_repair/v6_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v6_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v62_config.py",
    "external_agents_eval/workflows/solve_review_repair/v62_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v63_config.py",
)
PLACEHOLDER_LEDGER = {
    # Preflight only: proves Solver injection and review handoff without an A call.
    "summary": "Zero-API preflight placeholder; never used by a scored run.",
    "obligations": [], "preservation_obligations": [], "open_questions": [],
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES}


def runtime_manifest(archive):
    archive = Path(archive).resolve()
    manifest = json.loads((archive.parent / "runtime.json").read_text())
    if (manifest["archive"] != str(archive) or file_sha256(archive) != manifest["archive_sha256"]
            or manifest["pi_version"] != "0.84.2" or manifest["node_version"] != "v22.19.0"
            or manifest.get("rg_fd_included") is not False):
        raise ValueError("Pi runtime archive does not match its manifest")
    return {**manifest, "manifest_sha256": file_sha256(archive.parent / "runtime.json")}


# ---------------------------------------------------------------------------
# prepare / validate
# ---------------------------------------------------------------------------

def prepare(root, m0_root, archive, task_ids=None, note=None):
    """``task_ids`` selects a separately named recovery subset; it never merges into another root."""
    import yaml

    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    readiness = read(m0_root / "preflight/readiness.json")
    if not readiness["passed"]:
        raise ValueError("The M0 official gold/control preflight did not pass")
    runtime = runtime_manifest(archive)
    if task_ids:
        known = {task["task_id"] for task in read(m0_root / "manifests/public-tasks.json")}
        if not set(task_ids) <= known - set(EXCLUDED):
            raise ValueError("Unknown or excluded task in the subset: "
                             + ", ".join(sorted(set(task_ids) - (known - set(EXCLUDED)))))
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    copies = {"m0_run.py": "run.py", "native_compat.py": "native_compat.py",
              "sanitize_repository.py": "sanitize_repository.py"}
    for target, source in copies.items():
        shutil.copyfile(m0_root / "scripts" / source, root / "scripts" / target)
    (root / "manifests").mkdir()
    for name in ("public-tasks.json", "eval30_current.json", "task-file-sha256.json",
                 "heldout30.json", "heldout30.ERRATA.md"):
        if (m0_root / "manifests" / name).exists():
            shutil.copyfile(m0_root / "manifests" / name, root / "manifests" / name)
    (root / "grader").symlink_to(m0_root / "grader", target_is_directory=True)
    (root / "configs").mkdir()  # M0's prepare_config writes here
    native = load_native(root)
    tasks = native.load_tasks(root)
    config = native.prepare_config(root)
    original = yaml.safe_load((m0_root / "configs/native-deepseek-v4.1-flash.yaml").read_text())
    if config != original:
        raise ValueError("Mini prompts, tools, model parameters, or budgets differ from M0")
    rows = []
    for task in tasks:
        tid = task["task_id"]
        if tid in EXCLUDED or (task_ids and tid not in task_ids):
            continue
        output = m0_root / "outputs" / tid
        image = read(output / "image.json")
        m0 = read(output / "result.json")
        rows.append({
            "task_id": tid, "task": task,
            "image": {"id": image["id"], "repo_digests": image["repo_digests"]},
            "m0_status": m0["status"], "m0_reward": m0.get("reward"),
            "m0_result_sha256": file_sha256(output / "result.json"),
        })
    selection = {
        "experiment_id": EXPERIMENT_ID, "m0_root": str(m0_root),
        "task_count": len(rows), "excluded": EXCLUDED, "tasks": rows,
        "task_subset": sorted(task_ids) if task_ids else None, "note": note,
        "native_source_sha256": {name: file_sha256(root / "scripts" / name) for name in copies},
        "code_sha256": code_hashes(), "pi_runtime": runtime,
        "official_preflight": str(m0_root / "preflight/readiness.json"),
        "pipeline": [
            "A: Pi 0.84.2 exploration with the historical spec prompt and read/grep/find/ls "
            "(PI_OFFLINE=1, no rg/fd, filtered egress to the model API only)",
            "A: observed evidence frozen; one tool-free FT request (DeepSeek V4.1 Flash, "
            "thinking, high, max_tokens 65536, seed 42)",
            "Solver: native Mini M0 config + frozen ledger appended by ft_m1 augment_task",
            "P0: official verifier in a fresh container (analysis only)",
            "Review: ft_m1_review review_case (FrozenMiniReviewAgent, V6.3) -> P1 official verifier",
        ],
        "evidence_packet_to_solver": False, "images_removed_by_this_run": False,
    }
    save(root / "selection.json", selection)
    save(root / "preflight/plan.json", {
        "experiment_id": EXPERIMENT_ID, "task_count": len(rows), "workers_default": 3,
        "model_api_calls": 0, "model": "DeepSeek V4.1 Flash (deepseek-flash), all stages",
        "a_role_timeout_sec": A_ROLE_TIMEOUT_SEC, "solver": config["agent"],
    })
    return selection


def validate(root, native):
    import yaml

    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong experiment")
    for name, digest in selection["native_source_sha256"].items():
        if file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Frozen native source changed: " + name)
    if code_hashes() != selection["code_sha256"]:
        changed = [k for k, v in code_hashes().items() if selection["code_sha256"].get(k) != v]
        raise ValueError("Frozen experiment code changed: " + ", ".join(changed))
    runtime_manifest(selection["pi_runtime"]["archive"])
    native.load_tasks(root)
    config = native.prepare_config(root)
    m0 = Path(selection["m0_root"])
    if config != yaml.safe_load((m0 / "configs/native-deepseek-v4.1-flash.yaml").read_text()):
        raise ValueError("Mini configuration differs from M0")
    for row in selection["tasks"]:
        if file_sha256(m0 / "outputs" / row["task_id"] / "result.json") != row["m0_result_sha256"]:
            raise ValueError("An M0 reference result changed")
    return selection, config


def require_image(client, row):
    """This run never pulls or removes images; every held-out image is pinned locally."""
    image = client.images.get(row["image"]["id"])
    if image.id != row["image"]["id"] or row["image"]["repo_digests"][0] not in image.attrs.get(
            "RepoDigests", []):
        raise ValueError("Local task image differs from the frozen M0 image")
    return image


def ensure_image(native, row):
    # ft_m1_review.review_case removes images it pulled itself; keep that path unreachable.
    client = native.docker.from_env(timeout=60)
    try:
        require_image(client, row)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# A container: Pier semantics with filtered inference egress
# ---------------------------------------------------------------------------

def put_file(container, directory, name, data, mode=0o644):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        item = tarfile.TarInfo(name)
        item.size, item.mode = len(data), mode
        archive.addfile(item, io.BytesIO(data))
    if not container.put_archive(directory, stream.getvalue()):
        raise RuntimeError("Container upload failed: " + directory + "/" + name)


def container_checked(container, command, *, env=None, timeout=300):
    """Combined output, like Mini's execute used by M0's prepare_repository."""
    result = container.exec_run(
        ["timeout", "--signal=TERM", "--kill-after=5", str(timeout), "bash", "-c", command],
        workdir="/app", environment=env or None,
    )
    output = (result.output or b"").decode("utf-8", errors="replace")
    if result.exit_code:
        raise RuntimeError("A sandbox command failed: " + output[-2500:])
    return output


class PiContainerEnvironment(IsolatedEnvironment):
    """The BaseEnvironment surface PiRoleRunner needs, with Pier's exec semantics."""

    def __init__(self, container, *, instance_id, proxy_env):
        super().__init__(container, instance_id=instance_id)
        self.proxy_env = dict(proxy_env)

    def agent_process_env(self, values):
        # Pier's filtered-egress DockerEnvironment scopes proxy variables to agent commands.
        merged = dict(self.proxy_env)
        merged.update(values or {})
        return merged

    def exec_sync(self, command, *, cwd=None, env=None, timeout_sec=None, user=None, **kwargs):
        timeout = str(max(1, int(timeout_sec or DEFAULT_EXEC_TIMEOUT_SEC)))
        result = self.container.exec_run(
            ["timeout", "--signal=TERM", "--kill-after=5", timeout, "bash", "-c", command],
            workdir=cwd, environment=env or None, user="" if user is None else str(user),
            demux=True,
        )
        stdout, stderr = result.output or (b"", b"")
        return SimpleNamespace(
            return_code=result.exit_code,
            stdout=(stdout or b"").decode("utf-8", errors="replace"),
            stderr=(stderr or b"").decode("utf-8", errors="replace"),
        )


class ASandbox:
    """Internal network + allowlist proxy + task container; removed after A."""

    def __init__(self, client, image, task, allowlist):
        self.client, self.network, self.proxy, self.container = client, None, None, None
        self.token = secrets.token_urlsafe(24)
        self.allowlist = tuple(allowlist)
        suffix = uuid.uuid4().hex[:12]
        labels = {**LABEL, "task": task["task_id"], "stage": "A"}
        try:
            self.network = client.networks.create(
                "v63h-a-" + suffix, driver="bridge", internal=True, labels=labels)
            self.proxy = client.containers.create(
                image.id, entrypoint=[], command=["sh", "-c", PROXY_COMMAND],
                name="v63h-egress-" + suffix, network="bridge",
                environment={"PROXY_TOKEN": self.token, "PROXY_PORT": str(EGRESS_PORT),
                             "ALLOWLIST_DOMAINS": ",".join(self.allowlist)},
                cap_drop=["ALL"], security_opt=["no-new-privileges:true"], mem_limit="512m",
                pids_limit=256, labels={**labels, "role": "egress-proxy"},
            )
            put_file(self.proxy, "/tmp", "v63h_egress_proxy.py", PROXY_SCRIPT.read_bytes())
            self.network.connect(self.proxy, aliases=[EGRESS_ALIAS])
            self.proxy.start()
            deadline = time.monotonic() + 60
            while b"READY" not in self.proxy.logs():
                self.proxy.reload()
                if self.proxy.status != "running" or time.monotonic() > deadline:
                    raise RuntimeError("Egress proxy did not start: "
                                       + self.proxy.logs().decode(errors="replace")[-1000:])
                time.sleep(0.5)
            self.container = client.containers.create(
                image.id, entrypoint=[], command=["sleep", "infinity"],
                name="v63h-a-" + suffix, working_dir="/app", network=self.network.name,
                cap_drop=["ALL"], security_opt=["no-new-privileges:true"], pids_limit=512,
                nano_cpus=int(task["cpus"]) * 1_000_000_000,
                mem_limit=str(task["memory_mb"]) + "m", labels=labels,
            )
            self.container.start()
        except Exception:
            self.close(None)
            raise

    @property
    def proxy_env(self):
        # pier.environments.agent_setup.proxy_environment
        url = "http://agent:{}@{}:{}".format(self.token, EGRESS_ALIAS, EGRESS_PORT)
        return {"HTTP_PROXY": url, "HTTPS_PROXY": url, "http_proxy": url, "https_proxy": url,
                "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1"}

    def redact(self, text):
        return text.replace(self.token, "[REDACTED]")

    def close(self, output):
        if self.proxy is not None and output is not None:
            try:
                (output / "egress-proxy.log").write_text(
                    self.redact(self.proxy.logs().decode(errors="replace")))
            except Exception as error:  # noqa: BLE001 - keep cleaning up
                (output / "egress-proxy.log").write_text("log unavailable: " + str(error))
        for item in (self.container, self.proxy):
            if item is not None:
                try:
                    item.remove(force=True)
                except Exception:  # noqa: BLE001 - already gone
                    pass
        if self.network is not None:
            try:
                self.network.remove()
            except Exception:  # noqa: BLE001 - already gone
                pass


EGRESS_PROBE = r"""python - <<'PY'
import base64, json, os, socket
def direct_blocked():
    s = socket.socket(); s.settimeout(3)
    try:
        s.connect(("1.1.1.1", 443)); return False
    except OSError:
        return True
    finally:
        s.close()
def connect(target, auth):
    s = socket.create_connection(("%(alias)s", %(port)d), timeout=20)
    try:
        head = "CONNECT {0} HTTP/1.1\r\nHost: {0}\r\n".format(target)
        if auth:
            token = base64.b64encode(("agent:" + os.environ["PROXY_TOKEN"]).encode()).decode()
            head += "Proxy-Authorization: Basic " + token + "\r\n"
        s.sendall((head + "\r\n").encode())
        return s.recv(256).decode("latin-1").split("\r\n")[0]
    finally:
        s.close()
print(json.dumps({
    "direct_blocked": direct_blocked(),
    "allowed": connect("%(host)s:443", True),
    "other_host": connect("github.com:443", True),
    "other_port": connect("%(host)s:80", True),
    "no_auth": connect("%(host)s:443", False),
    "hidden_tests_present": os.path.exists("/tests/grader.py"),
    "solution_present": os.path.exists("/solution/solution.patch"),
}))
PY"""


def check_egress(probe):
    return (probe["direct_blocked"] and probe["allowed"].startswith("HTTP/1.1 200")
            and probe["other_host"].startswith("HTTP/1.1 403")
            and probe["other_port"].startswith("HTTP/1.1 403")
            and probe["no_auth"].startswith("HTTP/1.1 407")
            and not probe["hidden_tests_present"] and not probe["solution_present"])


def prepare_a_container(native, root, sandbox, task, output, runtime):
    """M0's prepare_repository steps; its network=none audit becomes the egress audit."""
    container = sandbox.container
    environment_patch = container_checked(container, native.snapshot_command(task["base_commit"]))
    (output / "environment-baseline.patch").write_text(environment_patch)
    native.upload(container, root / "scripts/sanitize_repository.py",
                  "/tmp/native_sanitize_repository.py")
    raw = container_checked(container, "python /tmp/native_sanitize_repository.py --repository /app "
                            "--revision " + shlex.quote(task["base_commit"]))
    audit = json.loads(raw.splitlines()[-1])
    container_checked(container, "git clean -fd -- .")
    if environment_patch:
        native.upload(container, environment_patch.encode(), "/tmp/native_environment.patch")
        container_checked(container, "git apply --binary /tmp/native_environment.patch && "
                          "rm /tmp/native_environment.patch")
    container_checked(container, "git config core.hooksPath /dev/null && git branch main HEAD")
    if container_checked(container, native.snapshot_command(audit["synthetic_commit"])) != environment_patch:
        raise RuntimeError("Git isolation changed the image's runnable baseline")
    scan = container_checked(container, "find /app /root /opt -name .git -type d -prune 2>/dev/null || true")
    others = [value for value in scan.splitlines() if value != "/app/.git"]
    for directory in others:
        container_checked(container, "rm -rf -- " + shlex.quote(directory))
    probe = json.loads(container_checked(
        container, EGRESS_PROBE % {"alias": EGRESS_ALIAS, "port": EGRESS_PORT,
                                   "host": sandbox.allowlist[0]},
        env={"PROXY_TOKEN": sandbox.token}).splitlines()[-1])
    container.reload()
    host = container.attrs["HostConfig"]
    networks = sorted(container.attrs["NetworkSettings"]["Networks"])
    if (not check_egress(probe) or networks != [sandbox.network.name] or host.get("Binds")
            or host.get("Privileged")):
        raise RuntimeError("A sandbox isolation audit failed: " + json.dumps(probe))
    # Pi runtime and Pier's PiPierAgent.setup, after the .git scan so nothing is removed.
    data = Path(runtime["archive"]).read_bytes()
    if sha256_bytes(data) != runtime["archive_sha256"]:
        raise RuntimeError("Pi runtime archive changed")
    container_checked(container, "mkdir -p /opt/pi-runtime")
    if not container.put_archive("/opt/pi-runtime", data):
        raise RuntimeError("Pi runtime upload failed")
    settings = json.dumps({"defaultProjectTrust": "never", "telemetry": False})
    container_checked(container, "mkdir -p /external_eval/pi_config /external_eval/home && "
                      "printf '%s' " + shlex.quote(settings) + " > /external_eval/pi_config/settings.json"
                      " && chmod -R 777 /external_eval")
    version = container_checked(container, "/opt/pi-runtime/bin/node " + PI_CLI + " --version").strip()
    if version.splitlines()[-1] != runtime["pi_version"]:
        raise RuntimeError("Pi runtime did not start: " + version[-500:])
    tools = container_checked(container, "PATH=" + PI_ENV["PATH"] + " command -v rg fd || true").split()
    audit.update(
        network="internal bridge + authenticated allowlist proxy", network_names=networks,
        egress_allowlist=list(sandbox.allowlist), egress_probe=probe,
        hidden_tests_present=False, solution_present=False,
        other_git_directories_removed=len(others),
        environment_patch_sha256=hashlib.sha256(environment_patch.encode()).hexdigest(),
        environment_patch_preserved=True, main_alias="synthetic baseline only",
        cpus=host["NanoCpus"] / 1_000_000_000, memory_bytes=host["Memory"],
        host_mounts=host.get("Binds") or [], pi_version=version.splitlines()[-1],
        pi_runtime_archive_sha256=runtime["archive_sha256"], rg_fd_on_path=tools,
    )
    save(output / "isolation.json", audit)
    return audit


def a_stage_id(task_id, base_commit, instruction, baseline_sha256, model_config, config):
    # PiV6SolveReviewRepairPierAgent.run with workflow_version "6.2" and no fork plan.
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"v6.2-spec:{task_id}:{base_commit}:"
        f"{sha256_bytes(instruction.encode('utf-8'))}:{baseline_sha256}:"
        f"{model_config.digest}:{config.workflow_prompt_version}:"
        f"{config.workflow_controller_version}:{config.prompt_version}:{config.controller_version}",
    ).hex


async def build_ledger(sandbox, task, output, api_key):
    """PiV62SolveReviewRepairPierAgent.run up to prepare_ledger, unchanged."""
    tid, instruction = task["task_id"], task["instruction"]
    store_root = output / "v6_review_repair"
    model_config = PiStageModelConfig.from_value(STAGE_MODEL_CONFIG)
    store = ArtifactStore(store_root, secrets=(api_key, sandbox.token))

    def redact(text):
        value = "" if text is None else str(text)
        return sandbox.redact(value.replace(api_key, "[REDACTED]"))

    environment = PiContainerEnvironment(sandbox.container, instance_id=tid,
                                         proxy_env=sandbox.proxy_env)
    env = environment.agent_process_env(
        {key: api_key for key in model_config.required_env_keys()})
    env.update(PI_ENV)
    runner = PiRoleRunner(
        environment=environment, logs_root=store_root, model=model_config.default.model,
        provider=model_config.default.provider, stage_model_config=model_config,
        agent_version=get_agent_spec("pi").upstream_ref, env=env, redact=redact,
    )
    workspace = ContainerGitWorkspaceManager(
        environment=environment, transfer_dir=store_root / ".transfer", repository="/app",
        redact=redact,
    )
    head = await environment.exec("git rev-parse --verify 'HEAD^{commit}'", cwd="/app")
    if head.return_code:
        raise RuntimeError("Cannot resolve the sanitized A baseline commit")
    base_commit = head.stdout.strip()
    try:
        environment_patch = await workspace.capture_patch(worktree="/app", base_commit=base_commit)
        baseline_sha256 = sha256_bytes(environment_patch)
        baseline_manifest = await workspace.baseline_manifest(
            worktree="/app", base_commit=base_commit, patch=environment_patch)
        store.write_bytes("inputs/environment.patch", environment_patch)
        store.write_json("inputs/environment-baseline.json", baseline_manifest)
        store.event("V6_FINAL_WORKTREE_BASELINE", patch_sha256=baseline_sha256,
                    had_changes=bool(environment_patch), baseline_manifest=baseline_manifest)
        paths = await workspace.prepare(base_commit=base_commit)
        for worktree in dict.fromkeys(path for path in (
                paths.solver, paths.spec_review, paths.patch_review, paths.repair, paths.summary)
                if path):
            if environment_patch:
                await workspace.apply_patch(worktree=worktree, patch=environment_patch)
        task_input = TaskInput(
            task_id=tid, issue_title=issue_title_from_instruction(instruction, tid),
            issue_body=instruction, repository_name=tid.split("-", 1)[0],
            base_commit=base_commit, managed_delivery=managed_delivery_from_instruction(instruction),
        )
        config = V62Config(role_timeout_sec=A_ROLE_TIMEOUT_SEC,
                           model=model_config.for_role("solver").model)
        a_model = model_config.for_role("A")
        controller = V62Controller(
            ledger_client=FTLedgerClient(model=a_model.model, base_url=a_model.effective_base_url,
                                         api_key=api_key),
            role_runner=runner, workspace=workspace, artifacts=store, config=config,
        )
        ledger = await controller.prepare_ledger(
            task_input, base_worktree=paths.spec_review, baseline_patch=environment_patch,
            stage_id=a_stage_id(tid, base_commit, instruction, baseline_sha256, model_config, config),
        )
        status = {
            "base_commit": base_commit, "baseline_patch_sha256": baseline_sha256,
            "ledger_quality": controller.ledger_quality,
            "ledger_warnings": list(controller.ledger_warnings),
            "ledger_error": controller.ledger_error,
        }
        if ledger is None:
            return None, status
        envelope = read(store_root / "v6/pre-solver-ledger.json")
        if envelope["ledger"] != ledger:
            raise RuntimeError("Pre-solver ledger envelope differs from the returned ledger")
        return envelope, status
    finally:
        await workspace.cleanup()


def run_a(native, root, row, output, api_key, runtime, ledger_relative=None):
    """Fresh A for one task. An existing partial attempt is never resampled."""
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("An A attempt already exists without a terminal record; not resampled")
    output.mkdir(parents=True, exist_ok=True)
    started = native.now()
    client = native.docker.from_env(timeout=1800)
    sandbox = None
    try:
        image = require_image(client, row)
        allowlist = PiStageModelConfig.from_value(STAGE_MODEL_CONFIG).domains()
        sandbox = ASandbox(client, image, row["task"], allowlist)
        isolation = prepare_a_container(native, root, sandbox, row["task"], output, runtime)
        envelope, status = asyncio.run(build_ledger(sandbox, row["task"], output, api_key))
    finally:
        if sandbox is not None:
            sandbox.close(output)
        client.close()
    workflow = output / "v6_review_repair"
    ft = workflow / "v6.2/ledger/result.json"
    record = {
        "task_id": row["task_id"], "started_at": started, "finished_at": native.now(),
        **status, "isolation_synthetic_commit": isolation["synthetic_commit"],
        "ft_result": {key: value for key, value in read(ft).items()
                      if key in {"status", "http_status", "usage", "attempts", "messages_sha256"}}
        if ft.exists() else None,
        "evidence_sha256": sha256_json(read(workflow / "v6.2/evidence.json"))
        if (workflow / "v6.2/evidence.json").exists() else None,
    }
    if envelope is None:
        record["status"] = "ledger_unavailable"
        save(output / "a-result.json", record)
        return record
    ledger = envelope["ledger"]
    relative = ledger_relative or "inputs/ledgers/" + row["task_id"] + ".json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    if ledger_input.canonical_sha256(ledger) != sha256_json(ledger):
        raise RuntimeError("Ledger hash conventions differ")
    record.update(
        status="ledger_ready", ledger_file=relative, ledger_file_sha256=file_sha256(path),
        ledger_sha256=sha256_json(ledger), envelope_ledger_sha256=envelope.get("ledger_sha256"),
        counts={name: len(ledger[name]) for name in
                ("obligations", "preservation_obligations", "open_questions")},
    )
    save(output / "a-result.json", record)
    return record


# ---------------------------------------------------------------------------
# Solver (ft_m1) and review (ft_m1_review)
# ---------------------------------------------------------------------------

def solver_input(root, row, a_record):
    ledger = ledger_input.load_frozen_ledger(root, {
        "task_id": row["task_id"], "ledger_file": a_record["ledger_file"],
        "ledger_file_sha256": a_record["ledger_file_sha256"],
        "ledger_sha256": a_record["ledger_sha256"],
    })
    augmented = ledger_input.augment_task(row["task"], ledger)
    # Pin the exact M0 image, without changing its contents (as ft_m1 does).
    augmented["image"] = row["image"]["repo_digests"][0]
    return augmented, ledger


def run_solver(native, root, row, a_record, config):
    """M0's per-task solve-then-grade loop on the ledger-augmented task (ft_m1)."""
    tid = row["task_id"]
    output = root / "solver" / tid
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("A Solver attempt already exists without a result; not resampled")
    augmented, ledger = solver_input(root, row, a_record)
    save(root / "inputs" / tid / "task.json", augmented)
    save(output / "input-audit.json", {
        "experiment_id": EXPERIMENT_ID, "task_id": tid, "task_checksum": row["task"]["task_checksum"],
        "original_instruction_sha256": sha256_bytes(row["task"]["instruction"].encode()),
        "augmented_instruction_sha256": sha256_bytes(augmented["instruction"].encode()),
        "ledger_sha256": a_record["ledger_sha256"], "ledger_file_sha256": a_record["ledger_file_sha256"],
        "a_output": str(root / "a" / tid),
        "counts": {name: len(ledger[name]) for name in
                   ("obligations", "preservation_obligations", "open_questions")},
        "evidence_packet_included": False, "review_guidance_included": False,
        "ledger_modified": False,
    })
    client = native.docker.from_env(timeout=1800)
    try:
        require_image(client, row)
        image, added = native.get_image(client, augmented, output)
        if added or image.id != row["image"]["id"]:
            raise RuntimeError("Solver image is not the pinned M0 image")
        model_patch, info = native.generate(augmented, root, output, config, image, "generate")
        result = native.grade(root, augmented, image.id, model_patch, output / "evaluation")
        result.update(exit_status=info.get("exit_status"),
                      patch_sha256=sha256_bytes(model_patch.encode()), finished_at=native.now(),
                      native_agent_completed=info.get("exit_status") == "Submitted")
    except Exception as error:  # noqa: BLE001 - retain per-task stage/infrastructure outcomes
        result = {"status": "infrastructure_or_stage_error", "task_id": tid,
                  "error_type": type(error).__name__, "error": str(error), "reward": None,
                  "finished_at": native.now()}
    finally:
        client.close()
    save(output / "result.json", result)
    return result


def review_row(root, row, a_record, solver_result):
    return {
        "task_id": row["task_id"], "task": row["task"], "image": row["image"],
        "source_root": str(root), "m1_output": str(root / "solver" / row["task_id"]),
        "ledger_file": str(root / a_record["ledger_file"]),
        "ledger_sha256": a_record["ledger_sha256"],
        "p0_sha256": solver_result["patch_sha256"], "m1_reward": solver_result.get("reward"),
    }


def run_case(native, root, row, config, runtime, update):
    tid = row["task_id"]
    api_key = os.environ["DEEPSEEK_API_KEY"]
    stage = "A"
    try:
        a_path = root / "a" / tid / "a-result.json"
        if a_path.exists():
            a_record = read(a_path)
        else:
            update(tid, "A")
            a_record = run_a(native, root, row, root / "a" / tid, api_key, runtime)
        if a_record["status"] != "ledger_ready":
            return {"task_id": tid, "status": "ledger_unavailable", "stage": "A", "reward": None,
                    "ledger_quality": a_record.get("ledger_quality"),
                    "ledger_error": a_record.get("ledger_error"), "finished_at": native.now()}
        stage = "Solver"
        s_path = root / "solver" / tid / "result.json"
        if s_path.exists():
            solver_result = read(s_path)
        else:
            update(tid, "solving_then_p0_score", ledger_quality=a_record["ledger_quality"])
            solver_result = run_solver(native, root, row, a_record, config)
        if solver_result["status"] == "infrastructure_or_stage_error":
            return {**solver_result, "stage": "Solver"}
        stage = "review"
        update(tid, "I/R_S_repair_then_verifier", p0_reward=solver_result.get("reward"))
        ensure_image(native, row)
        result = review.review_case(native, root, review_row(root, row, a_record, solver_result))
        result.update(experiment_id=EXPERIMENT_ID, ledger_quality=a_record["ledger_quality"],
                      solver_exit_status=solver_result.get("exit_status"),
                      p0_status=solver_result.get("status"), m0_reward=row["m0_reward"])
        return result
    except Exception as error:  # noqa: BLE001 - archive per-task stage/infrastructure failures
        return {"task_id": tid, "status": "infrastructure_or_stage_error", "stage": stage,
                "reward": None, "error_type": type(error).__name__, "error": str(error),
                "finished_at": native.now()}


def run_batch(root, selection, native, config, workers):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    readiness = read(root / "preflight/readiness.json")
    if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
        raise ValueError("Run the zero-API preflight for this frozen selection first")
    runtime = selection["pi_runtime"]
    lock = threading.Lock()
    states = {row["task_id"]: {"phase": "queued"} for row in selection["tasks"]}
    results = {}

    def update(tid, phase, **values):
        with lock:
            states[tid] = {**states[tid], "phase": phase, "updated_at": native.now(), **values}
            save(root / "monitor-status.json", {"workers": workers, "tasks": states})

    def one(row):
        tid = row["task_id"]
        path = root / "outputs" / tid / "result.json"
        result = read(path) if path.exists() else run_case(native, root, row, config, runtime, update)
        save(path, result)
        with lock:
            results[tid] = result
        update(tid, result["status"], reward=result.get("reward"), p0_reward=result.get("p0_reward"))
        with lock:
            save(root / "result.json", {
                "experiment_id": EXPERIMENT_ID, "task_count": len(states),
                "terminal_count": len(results),
                "passed": sum(r.get("reward") == 1 for r in results.values()),
                "valid_failed": sum(r.get("reward") == 0 for r in results.values()),
                "results": list(results.values()), "updated_at": native.now(),
            })
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in selection["tasks"]]):
            future.result()
    save(root / "launcher-result.json", {"completed": True, "terminal_count": len(results),
                                         "finished_at": native.now()})


# ---------------------------------------------------------------------------
# zero-API preflight and the A-only smoke
# ---------------------------------------------------------------------------

def preflight(root, selection, native, config):
    official = read(selection["official_preflight"])
    if not official["passed"]:
        raise ValueError("The M0 official gold/control preflight is not valid")
    client = native.docker.from_env(timeout=1800)
    try:
        images = {row["task_id"]: require_image(client, row).id for row in selection["tasks"]}
    finally:
        client.close()
    row = selection["tasks"][0]
    tid = row["task_id"]
    output = root / "preflight" / tid
    runtime = selection["pi_runtime"]
    # 1. A sandbox: repository isolation, filtered egress, Pi runtime and the actual
    #    PiRoleRunner path. An invalid credential is rejected by the API: no model call.
    a_output = output / "a-sandbox"
    a_output.mkdir(parents=True, exist_ok=True)
    client = native.docker.from_env(timeout=1800)
    sandbox = None
    try:
        image = require_image(client, row)
        allowlist = PiStageModelConfig.from_value(STAGE_MODEL_CONFIG).domains()
        sandbox = ASandbox(client, image, row["task"], allowlist)
        isolation = prepare_a_container(native, root, sandbox, row["task"], a_output, runtime)
        pi_probe = asyncio.run(probe_pi(sandbox, row["task"], a_output))
    finally:
        if sandbox is not None:
            sandbox.close(a_output)
        client.close()
    proxy_log = (a_output / "egress-proxy.log").read_text()
    # The egress probe opens one allowed tunnel; Pi's request must open another. Pi's JSON
    # mode exits 0 on an API error, so the rejection is read from its transcript.
    pi_reached_api = (proxy_log.count(f"ALLOW CONNECT {allowlist[0]}:443") >= 2
                      and pi_probe["authentication_rejected"])
    # 2. Solver input: the native Mini user message carries exactly one ledger supplement.
    placeholder = output / "placeholder-ledger.json"
    placeholder.write_text(json.dumps(PLACEHOLDER_LEDGER, ensure_ascii=False, indent=2) + "\n")
    augmented = ledger_input.augment_task(row["task"], PLACEHOLDER_LEDGER)
    augmented["image"] = row["image"]["repo_digests"][0]
    client = native.docker.from_env(timeout=1800)
    try:
        image, added = native.get_image(client, augmented, output / "solver-offline")
        if added or image.id != row["image"]["id"]:
            raise ValueError("Preflight image differs from M0")
        _, protocol = native.generate(augmented, root, output / "solver-offline", config, image,
                                      "offline")
    finally:
        client.close()
    trajectory = read(output / "solver-offline/native.traj.json")
    user = next(m["content"] for m in trajectory["messages"] if m["role"] == "user")
    injected = (row["task"]["instruction"] in user and user.count(ledger_input.HEADING) == 1
                and user.count(ledger_input.GUIDANCE) == 1)
    # 3. Review handoff: fresh review container, P0 conversion and frozen fork (no model).
    p0 = (output / "solver-offline/model.patch").read_bytes()
    handoff = review.review_case(native, root, {
        "task_id": tid, "task": row["task"], "image": row["image"], "source_root": str(root),
        "m1_output": str(output / "solver-offline"), "ledger_file": str(placeholder),
        "ledger_sha256": sha256_json(PLACEHOLDER_LEDGER), "p0_sha256": sha256_bytes(p0),
        "m1_reward": None,
    }, preflight=True)
    passed = bool(isolation and pi_reached_api and injected and handoff["passed"])
    save(root / "preflight/readiness.json", {
        "passed": passed, "experiment_id": EXPERIMENT_ID, "model_api_calls": 0,
        "selection_sha256": sha256_json(selection), "task_id": tid,
        "images_verified": len(images),
        "a_sandbox": {"isolation": isolation, "pi_probe": pi_probe,
                      "pi_reached_model_api_through_proxy": pi_reached_api},
        "solver_offline_protocol": protocol, "ledger_supplement_injected_once": injected,
        "review_handoff": handoff,
        "official_gold_control_reused_from": selection["official_preflight"],
        "not_covered": "no real A/FT/Solver/review model call; see the A-only smoke",
        "finished_at": native.now(),
    })
    if not passed:
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


async def probe_pi(sandbox, task, output):
    """Run the real PiRoleRunner command path with an invalid key (auth fails, nothing billed)."""
    from external_agents_eval.workflows.solve_review_repair.models import RoleName

    model_config = PiStageModelConfig.from_value(STAGE_MODEL_CONFIG)
    environment = PiContainerEnvironment(sandbox.container, instance_id=task["task_id"],
                                         proxy_env=sandbox.proxy_env)
    env = environment.agent_process_env(
        {key: "preflight-invalid-key" for key in model_config.required_env_keys()})
    env.update(PI_ENV)
    runner = PiRoleRunner(
        environment=environment, logs_root=output / "pi-probe",
        model=model_config.default.model, provider=model_config.default.provider,
        stage_model_config=model_config, agent_version=get_agent_spec("pi").upstream_ref,
        env=env, redact=sandbox.redact,
    )
    result = await runner.run_role(
        role=RoleName.V5_SPECIFICATION_REVIEWER, prompt="Reply with OK.", worktree="/app",
        tools=("read", "grep", "find", "ls"), timeout_sec=180, artifact_prefix="probe",
    )
    stderr = output / "pi-probe/probe/stderr.txt"
    transcript = output / "pi-probe/probe/pi.jsonl"
    text = (stderr.read_text() if stderr.exists() else "") + (
        transcript.read_text() if transcript.exists() else "")
    return {"exit_code": result.exit_code, "timed_out": result.timed_out,
            "authentication_rejected": "401" in text and "authentication_error" in text,
            "command": _build_pi_command(model=model_config.default.model,
                                         provider=model_config.default.provider,
                                         prompt_path="<prompt>", tools=("read", "grep", "find", "ls"))}


def smoke_a(root, selection, native, task_id):
    """One real A (Pi + FT) in a separate, uncounted directory."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    readiness = read(root / "preflight/readiness.json")
    if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
        raise ValueError("Run the zero-API preflight first")
    row = next(row for row in selection["tasks"] if row["task_id"] == task_id)
    output = root / "smoke-a" / task_id
    record = run_a(native, root, row, output, os.environ["DEEPSEEK_API_KEY"],
                   selection["pi_runtime"], ledger_relative=f"smoke-a/{task_id}/ledger.json")
    save(output / "smoke.json", {"counted": False, "purpose": "A-stage smoke before the batch",
                                 **{k: record.get(k) for k in ("status", "ledger_quality", "counts",
                                                               "ft_result", "ledger_error")}})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "smoke-a", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--m0-root", type=Path)
    parser.add_argument("--pi-runtime", type=Path)
    parser.add_argument("--task-id", action="append",
                        help="prepare: restrict to these tasks (repeat); smoke-a: the one task")
    parser.add_argument("--note", help="prepare: why this subset run exists")
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if not args.m0_root or not args.pi_runtime:
            parser.error("prepare requires --m0-root and --pi-runtime")
        selection = prepare(root, args.m0_root.resolve(), args.pi_runtime.resolve(),
                            set(args.task_id or []), args.note)
    else:
        native = load_native(root)
        selection, config = validate(root, native)
        if args.mode == "preflight":
            preflight(root, selection, native, config)
        elif args.mode == "smoke-a":
            if not args.task_id or len(args.task_id) != 1:
                parser.error("smoke-a requires exactly one --task-id")
            smoke_a(root, selection, native, args.task_id[0])
        else:
            run_batch(root, selection, native, config, args.workers)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID,
                      "task_count": selection["task_count"]}))


if __name__ == "__main__":
    main()
