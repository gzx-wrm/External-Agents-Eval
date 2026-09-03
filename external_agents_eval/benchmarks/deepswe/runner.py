from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from urllib.parse import urlsplit

from external_agents_eval.agents.config import AGENTS, DEFAULT_MODEL
from external_agents_eval.benchmarks.deepswe.config import (
    CODEX_CONFIG_FILE,
    CODEX_MODEL_PROVIDER,
    CODEX_PROVIDER_BASE_URL,
    CODEX_VERSION,
    CODEX_WIRE_API,
    DEEPSWE_AGENTS,
)
from external_agents_eval.benchmarks.deepswe.manifest import (
    DeepSWEManifest,
    validate_manifest_against_checkout,
)
from external_agents_eval.benchmarks.deepswe.pier_compat import DEEPSWE_BUILDER_ENV
from external_agents_eval.benchmarks.swebench.manifest import atomic_write_json

AGENT_IMPORT_PATHS = {
    "pi": "external_agents_eval.agents.pier.pi:PiPierAgent",
    "hermes": "external_agents_eval.agents.pier.hermes:HermesPierAgent",
}
PIER_COMPAT_MODULE = "external_agents_eval.benchmarks.deepswe.pier_compat"
DOCKER_BUILDER_PREFIX = "external-eval-deepswe"


def _resolve_pier_executable(pier_executable: str) -> str | None:
    executable = shutil.which(pier_executable)
    if executable is not None or pier_executable != "pier":
        return executable
    # Keep the venv path intact: resolving the Python symlink first can jump
    # back to the base interpreter directory, where the Pier script is absent.
    sibling = Path(sys.executable).with_name("pier")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def default_job_name(agent_id: str, manifest: DeepSWEManifest) -> str:
    return f"deepswe-v1.1-{agent_id}-random{manifest.sample_size}-seed{manifest.sample_seed}"


def build_pier_command(
    *,
    pier_executable: str,
    checkout: Path,
    manifest: DeepSWEManifest,
    agent_id: str,
    jobs_dir: Path,
    job_name: str,
    model: str = DEFAULT_MODEL,
    num_workers: int = 1,
    attempts: int = 1,
    max_turns: int | None = None,
    environment: str = "docker",
) -> list[str]:
    if agent_id not in DEEPSWE_AGENTS:
        raise ValueError(f"Unsupported DeepSWE agent: {agent_id}")
    if num_workers <= 0 or attempts <= 0:
        raise ValueError("num_workers and attempts must be positive")
    if max_turns is not None and max_turns <= 0:
        raise ValueError("max_turns must be positive when provided")
    if environment not in {"docker", "modal"}:
        raise ValueError(f"Unsupported Pier environment: {environment}")
    command = [
        pier_executable,
        "run",
        "--path",
        str(checkout / "tasks"),
    ]
    if agent_id == "codex":
        command.extend(["--agent", "codex"])
    else:
        command.extend(["--agent-import-path", AGENT_IMPORT_PATHS[agent_id]])
    command.extend(
        [
            "--model",
            model,
            "--agent-env",
            "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}",
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            job_name,
            "--n-concurrent",
            str(num_workers),
            "--n-attempts",
            str(attempts),
            "--env",
            environment,
            "--yes",
        ]
    )
    if agent_id == "codex":
        command.extend(
            [
                "--agent-kwarg",
                f"version={CODEX_VERSION}",
                "--agent-kwarg",
                f"config_toml_file={CODEX_CONFIG_FILE.resolve()}",
            ]
        )
    if agent_id == "hermes" and max_turns is not None:
        command.extend(["--agent-kwarg", f"max_turns={max_turns}"])
    for task_id in manifest.task_ids:
        command.extend(["--include-task-name", task_id])
    return command


def _apply_docker_cleanup_compat(command: list[str], environment: str) -> list[str]:
    if environment != "docker":
        return command
    return [sys.executable, "-m", PIER_COMPAT_MODULE, *command[1:]]


def _docker_builder_name(jobs_dir: Path, job_name: str) -> str:
    identity = f"{jobs_dir.resolve()}\0{job_name}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:12]
    return f"{DOCKER_BUILDER_PREFIX}-{digest}"


def _docker_desktop_proxy_driver_options() -> list[str]:
    """Forward Docker Desktop's container-reachable proxy to BuildKit.

    Docker Desktop applies its proxy to daemon image pulls, but a BuildKit
    container created with the ``docker-container`` driver does not inherit
    that proxy.  In proxy/Fake-IP setups this makes the daemon pull succeed
    while BuildKit times out fetching registry auth tokens.

    Only ``*.docker.internal`` endpoints are forwarded.  Loopback proxies are
    not reachable from the builder container, and credential-bearing proxy
    URLs must not be copied into command-line arguments.
    """
    result = subprocess.run(
        ["docker", "info", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    try:
        docker_info = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        warnings.warn(
            "Could not parse Docker info while detecting the Docker Desktop proxy",
            stacklevel=2,
        )
        return []

    options: list[str] = []
    for info_key, environment_key in (
        ("HttpProxy", "http_proxy"),
        ("HttpsProxy", "https_proxy"),
    ):
        value = docker_info.get(info_key)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value.strip()
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        parsed = urlsplit(normalized)
        if parsed.username or parsed.password:
            continue
        hostname = (parsed.hostname or "").lower()
        if not hostname.endswith(".docker.internal"):
            continue
        options.extend(["--driver-opt", f"env.{environment_key}={normalized}"])

    no_proxy = docker_info.get("NoProxy")
    if options and isinstance(no_proxy, str) and no_proxy.strip():
        options.extend(["--driver-opt", f"env.no_proxy={no_proxy.strip()}"])
    return options


def _create_dedicated_builder(builder_name: str) -> None:
    """Create a BuildKit instance used only by one DeepSWE job."""
    inspect = subprocess.run(
        ["docker", "buildx", "inspect", builder_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0:
        # The job directory/run record guard prevents an active same-name job.
        # Therefore an existing deterministic builder is stale from an earlier
        # interrupted launch and is safe to replace.
        subprocess.run(
            ["docker", "buildx", "rm", "--force", builder_name],
            check=True,
        )
    create_command = [
        "docker",
        "buildx",
        "create",
        "--name",
        builder_name,
        "--driver",
        "docker-container",
    ]
    create_command.extend(_docker_desktop_proxy_driver_options())
    create_command.append("--bootstrap")
    subprocess.run(create_command, check=True)


def _remove_dedicated_builder(builder_name: str) -> None:
    """Best-effort removal of the job builder and all of its remaining cache."""
    try:
        result = subprocess.run(
            ["docker", "buildx", "rm", "--force", builder_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        warnings.warn(
            f"Failed to remove dedicated Docker builder {builder_name!r}: {error}",
            stacklevel=2,
        )
        return
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        warnings.warn(
            f"Failed to remove dedicated Docker builder {builder_name!r}: {detail}",
            stacklevel=2,
        )


def _run_pier_process(command: list[str], process_env: dict[str, str]) -> None:
    """Wait for Pier cleanup after the terminal sends both processes Ctrl-C.

    ``subprocess.run`` kills its child as soon as the parent receives
    ``KeyboardInterrupt``. Pier needs a little longer to cancel active trials
    and run Docker ``down --rmi all``. Waiting on the already-signalled child
    preserves that cleanup path. A second Ctrl-C still aborts immediately.
    """
    process = subprocess.Popen(command, env=process_env)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            raise
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_pier(
    *,
    checkout: Path,
    manifest: DeepSWEManifest,
    agent_id: str,
    jobs_dir: Path,
    job_name: str | None = None,
    model: str = DEFAULT_MODEL,
    num_workers: int = 1,
    attempts: int = 1,
    max_turns: int | None = None,
    pier_executable: str = "pier",
    environment: str = "docker",
    dry_run: bool = False,
) -> Path:
    if agent_id not in DEEPSWE_AGENTS:
        raise ValueError(f"Unsupported agent: {agent_id}")
    if not os.environ.get("DEEPSEEK_API_KEY") and not dry_run:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Use scripts/run_with_deepseek_key.sh.")
    resolved_checkout = validate_manifest_against_checkout(manifest, checkout)
    executable = _resolve_pier_executable(pier_executable)
    if executable is None and not dry_run:
        raise RuntimeError(
            "Pier is not installed. Use Python 3.12+ and install the 'deepswe' extra."
        )
    executable = executable or pier_executable
    resolved_jobs_dir = jobs_dir.expanduser().resolve()
    resolved_name = job_name or default_job_name(agent_id, manifest)
    job_dir = resolved_jobs_dir / resolved_name
    command = build_pier_command(
        pier_executable=executable,
        checkout=resolved_checkout,
        manifest=manifest,
        agent_id=agent_id,
        jobs_dir=resolved_jobs_dir,
        job_name=resolved_name,
        model=model,
        num_workers=num_workers,
        attempts=attempts,
        max_turns=max_turns,
        environment=environment,
    )
    command = _apply_docker_cleanup_compat(command, environment)
    builder_name = (
        _docker_builder_name(resolved_jobs_dir, resolved_name) if environment == "docker" else None
    )
    metadata = {
        "benchmark": "deepswe",
        "benchmark_revision": "1.1",
        "agent": agent_id,
        "model": model,
        "environment": environment,
        "max_turns": max_turns if agent_id == "hermes" else None,
        "turn_limit_policy": (
            "agent-max-turns" if agent_id == "hermes" and max_turns is not None
            else "task-timeout-only"
        ),
        "manifest": str(manifest.path),
        "task_ids": list(manifest.task_ids),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pier_cleanup_compat": environment == "docker",
        "docker_builder": builder_name,
        "docker_builder_cache_policy": (
            "prune-after-each-trial-remove-after-job" if builder_name else None
        ),
        "command": command,
    }
    if agent_id == "codex":
        metadata.update(
            {
                "agent_version": CODEX_VERSION,
                "agent_implementation": "pier-native",
                "model_provider": CODEX_MODEL_PROVIDER,
                "model_provider_base_url": CODEX_PROVIDER_BASE_URL,
                "model_provider_wire_api": CODEX_WIRE_API,
            }
        )
    else:
        metadata["agent_upstream_commit"] = AGENTS[agent_id].upstream_commit
    if dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return job_dir

    resolved_jobs_dir.mkdir(parents=True, exist_ok=True)
    if job_dir.exists():
        raise ValueError(
            f"Pier job directory already exists: {job_dir}. Choose a new --job-name "
            "or use Pier's resume command directly."
        )
    run_record = resolved_jobs_dir / f"{resolved_name}.run.json"
    if run_record.exists():
        raise ValueError(f"Pier run record already exists: {run_record}. Choose a new --job-name.")
    process_env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[3]
    python_path = process_env.get("PYTHONPATH")
    process_env["PYTHONPATH"] = (
        str(project_root) if not python_path else f"{project_root}{os.pathsep}{python_path}"
    )
    if builder_name:
        process_env["BUILDX_BUILDER"] = builder_name
        process_env[DEEPSWE_BUILDER_ENV] = builder_name

    builder_created = False
    try:
        if builder_name:
            _create_dedicated_builder(builder_name)
            builder_created = True
        atomic_write_json(run_record, metadata)
        _run_pier_process(command, process_env)
    finally:
        if builder_created and builder_name:
            _remove_dedicated_builder(builder_name)
    return job_dir
