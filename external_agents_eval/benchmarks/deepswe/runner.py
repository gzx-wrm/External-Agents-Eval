from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from urllib.parse import urlsplit

from external_agents_eval.agents.config import AGENTS, DEFAULT_MODEL
from external_agents_eval.agents.pier.pi_model_config import (
    PiStageModelConfig,
    stage_config_json,
)
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
from external_agents_eval.benchmarks.deepswe.pier_compat import (
    DEEPSWE_BUILDER_ENV,
    _is_transient_environment_start_error,
    _patch_proxy_apt_transport,
)
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    ReviewerExperimentManifest,
    freeze_few_shots,
    git_object_ids_equivalent,
    load_frozen_few_shots,
    load_reviewer_experiment_manifest,
    task_base_commit,
)
from external_agents_eval.benchmarks.swebench.manifest import atomic_write_json
from external_agents_eval.workflows.solve_review_repair.v5_config import (
    V5_CONTROLLER_VERSION,
    V5_PROMPT_VERSION,
)
from external_agents_eval.workflows.solve_review_repair.v6_config import (
    V6_CONTROLLER_VERSION,
    V6_PROMPT_VERSION,
)
from external_agents_eval.workflows.solve_review_repair.v6_fork import V6ForkPlan, V6ForkPlans

AGENT_IMPORT_PATHS = {
    "pi": "external_agents_eval.agents.pier.pi:PiPierAgent",
    "pi-review-repair": (
        "external_agents_eval.agents.pier.pi_solve_review_repair:PiSolveReviewRepairPierAgent"
    ),
    "pi-v5-review-repair": (
        "external_agents_eval.agents.pier.pi_v5_solve_review_repair:PiV5SolveReviewRepairPierAgent"
    ),
    "pi-v62-review-repair": (
        "external_agents_eval.agents.pier.pi_v62_solve_review_repair:PiV62SolveReviewRepairPierAgent"
    ),
    "pi-v6-review-repair": (
        "external_agents_eval.agents.pier.pi_v6_solve_review_repair:PiV6SolveReviewRepairPierAgent"
    ),
    "hermes": "external_agents_eval.agents.pier.hermes:HermesPierAgent",
}
PIER_COMPAT_MODULE = "external_agents_eval.benchmarks.deepswe.pier_compat"
DOCKER_BUILDER_PREFIX = "external-eval-deepswe"
REVIEW_AGENT_TIMEOUT_MULTIPLIER = 4.0
REVIEW_ROLE_TIMEOUTS_SEC = {
    "solver": 10_800,
    "repair": 10_800,
    "specification_reviewer": 3_600,
    "patch_reviewer": 3_600,
    "json_corrector": 1_800,
}


def resolve_review_prompt_version(
    contract_version: str, review_arm: str, requested: str | None
) -> str:
    if requested is not None and requested != "4.0":
        raise ValueError(f"Unsupported reviewer prompt version: {requested}")
    if requested == "4.0" and (contract_version != "3.0" or review_arm not in {"E1", "E2"}):
        raise ValueError("Reviewer prompt v4.0 requires contract v3.0 and experiment arm E1 or E2")
    if requested is not None:
        return requested
    return "1.3" if contract_version == "1.0" else "2.3" if contract_version == "2.0" else "3.0"


def resolve_review_contract_version(review_arm: str, requested: str | None) -> str:
    if review_arm not in {"E0", "E1", "E2"}:
        raise ValueError(f"Unsupported reviewer experiment arm: {review_arm}")
    contract = requested or ("1.0" if review_arm == "E0" else "2.0")
    if contract not in {"1.0", "2.0", "3.0"}:
        raise ValueError(f"Unsupported reviewer contract version: {contract}")
    allowed = {"1.0"} if review_arm == "E0" else {"2.0", "3.0"}
    if contract not in allowed:
        raise ValueError(f"Reviewer arm {review_arm} is incompatible with contract {contract}")
    return contract


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
    stage_model_config: dict | str | None = None,
    num_workers: int = 1,
    attempts: int = 1,
    max_turns: int | None = None,
    environment: str = "docker",
    reviewer_manifest: ReviewerExperimentManifest | None = None,
    fork_plan: Path | None = None,
    fork_plans: Path | None = None,
    workflow_timeout_sec: int | None = None,
    few_shot_dir: Path | None = None,
    few_shot_digest: str | None = None,
    review_arm: str = "E0",
    review_contract_version: str | None = None,
    review_prompt_version: str | None = None,
    include_task_ids: tuple[str, ...] | None = None,
) -> list[str]:
    if agent_id not in DEEPSWE_AGENTS:
        raise ValueError(f"Unsupported DeepSWE agent: {agent_id}")
    if num_workers <= 0 or attempts <= 0:
        raise ValueError("num_workers and attempts must be positive")
    if max_turns is not None and max_turns <= 0:
        raise ValueError("max_turns must be positive when provided")
    if workflow_timeout_sec is not None and workflow_timeout_sec <= 0:
        raise ValueError("workflow_timeout_sec must be positive when provided")
    resolved_stage_models = PiStageModelConfig.from_value(
        stage_model_config,
        fallback_model=model,
    )
    if stage_model_config is not None and agent_id not in {
        "pi",
        "pi-v5-review-repair",
        "pi-v6-review-repair",
        "pi-v62-review-repair",
    }:
        raise ValueError("stage_model_config is valid only for Pi agents")
    if agent_id == "pi-review-repair":
        contract_version = resolve_review_contract_version(
            review_arm, review_contract_version
        )
        resolve_review_prompt_version(
            contract_version, review_arm, review_prompt_version
        )
    else:
        contract_version = None
        if review_arm != "E0":
            raise ValueError(
                "Reviewer experiment arms are valid only for pi-review-repair"
            )
        if review_contract_version is not None:
            raise ValueError(
                "Reviewer contract versions are valid only for pi-review-repair"
            )
        if review_prompt_version is not None:
            raise ValueError(
                "Reviewer prompt versions are valid only for pi-review-repair"
            )
    if reviewer_manifest is not None and agent_id not in {
        "pi-review-repair",
        "pi-v5-review-repair",
    }:
        raise ValueError("A reviewer manifest is valid only for a Pi reviewer workflow")
    if fork_plan is not None and fork_plans is not None:
        raise ValueError("Use either fork_plan or fork_plans")
    if (fork_plan is not None or fork_plans is not None) and agent_id not in {
        "pi-v6-review-repair", "pi-v62-review-repair",
    }:
        raise ValueError("Fork plans require pi-v6-review-repair or pi-v62-review-repair")
    if agent_id in {
        "pi-review-repair",
        "pi-v5-review-repair",
        "pi-v6-review-repair",
        "pi-v62-review-repair",
    } and attempts != 1:
        raise ValueError("The review workflow forbids automatic task-level retries")
    if (few_shot_dir is None) != (few_shot_digest is None):
        raise ValueError("few_shot_dir and few_shot_digest must be provided together")
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
    for env_key in resolved_stage_models.required_env_keys():
        command.extend(["--agent-env", f"{env_key}=${{{env_key}}}"])
    if stage_model_config is not None and agent_id in {
        "pi",
        "pi-v5-review-repair",
        "pi-v6-review-repair",
        "pi-v62-review-repair",
    }:
        command.extend(
            [
                "--agent-kwarg",
                f"stage_model_config={stage_config_json(resolved_stage_models)}",
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
    if reviewer_manifest is not None:
        command.extend(["--agent-kwarg", f"p0_manifest={reviewer_manifest.path.resolve()}"])
    if fork_plan is not None:
        command.extend(["--agent-kwarg", f"fork_plan={Path(fork_plan).expanduser().resolve()}"])
    if fork_plans is not None:
        command.extend(["--agent-kwarg", f"fork_plans={Path(fork_plans).expanduser().resolve()}"])
    if few_shot_dir is not None:
        command.extend(
            [
                "--agent-kwarg",
                f"few_shot_dir={few_shot_dir.resolve()}",
                "--agent-kwarg",
                f"few_shot_digest={few_shot_digest}",
            ]
        )
    if agent_id in {
        "pi-review-repair",
        "pi-v5-review-repair",
        "pi-v6-review-repair",
        "pi-v62-review-repair",
    } and workflow_timeout_sec is not None:
        command.extend(["--agent-kwarg", f"workflow_timeout_sec={workflow_timeout_sec}"])
    if agent_id == "pi-review-repair":
        assert contract_version is not None
        command.extend(["--agent-kwarg", f"review_arm={review_arm}"])
        command.extend(["--agent-kwarg", f"review_contract_version={contract_version}"])
        if review_prompt_version is not None:
            command.extend(["--agent-kwarg", f"review_prompt_version={review_prompt_version}"])
        command.extend(
            [
                "--agent-timeout-multiplier",
                f"{REVIEW_AGENT_TIMEOUT_MULTIPLIER:g}",
            ]
        )
    if agent_id in {"pi-v5-review-repair", "pi-v6-review-repair", "pi-v62-review-repair"}:
        command.extend(["--agent-timeout-multiplier", f"{REVIEW_AGENT_TIMEOUT_MULTIPLIER:g}"])
    available_task_ids = (
        reviewer_manifest.task_ids if reviewer_manifest is not None else manifest.task_ids
    )
    selected_task_ids = include_task_ids if include_task_ids is not None else available_task_ids
    if len(selected_task_ids) != len(set(selected_task_ids)):
        raise ValueError("Selected DeepSWE task IDs must be unique")
    unknown = sorted(set(selected_task_ids) - set(available_task_ids))
    if not selected_task_ids or unknown:
        detail = ": " + ", ".join(unknown) if unknown else ""
        raise ValueError("Selected DeepSWE tasks are empty or outside the active manifest" + detail)
    for task_id in selected_task_ids:
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


class EgressProxyPrewarmError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, retryable: bool) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable


def _prewarm_egress_proxy(builder_name: str, *, max_attempts: int = 3) -> int:
    """Populate the job builder with Pier's exact egress-proxy layers."""
    from pier.environments.agent_setup import (
        new_proxy_token,
        write_docker_proxy_compose,
    )
    from pier.models.agent.network import NetworkAllowlist

    with tempfile.TemporaryDirectory(prefix="external-eval-pier-proxy-") as temporary:
        root = Path(temporary)
        proxy_dir = root / "egress-proxy"
        write_docker_proxy_compose(
            path=root / "docker-compose-egress-proxy.json",
            proxy_dir=proxy_dir,
            allowlist=NetworkAllowlist(domains=["prewarm.invalid"]),
            token=new_proxy_token(),
        )
        _patch_proxy_apt_transport(proxy_dir / "Dockerfile")
        command = [
            "docker",
            "buildx",
            "build",
            "--builder",
            builder_name,
            "--progress=plain",
            str(proxy_dir),
        ]
        for attempt in range(1, max_attempts + 1):
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as error:
                raise EgressProxyPrewarmError(
                    f"Could not start the egress-proxy cache prewarm: {error}",
                    attempts=attempt,
                    retryable=False,
                ) from error
            if result.returncode == 0:
                return attempt
            output = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part
            )
            retryable = _is_transient_environment_start_error(output)
            if not retryable or attempt == max_attempts:
                detail = output[-4000:] or f"docker buildx exited with {result.returncode}"
                raise EgressProxyPrewarmError(
                    f"Egress-proxy cache prewarm failed: {detail}",
                    attempts=attempt,
                    retryable=retryable,
                )
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("egress-proxy prewarm loop did not return")


def _validate_reviewer_task_bases(
    *,
    checkout: Path,
    manifest: ReviewerExperimentManifest,
    task_ids: tuple[str, ...],
) -> None:
    """Fail before launch when frozen P0 metadata targets another task commit."""
    for task_id in task_ids:
        declared = task_base_commit(checkout / "tasks" / task_id)
        frozen = manifest.patches[task_id].base_commit
        if not git_object_ids_equivalent(frozen, declared):
            raise ValueError(
                f"P0 base commit for {task_id} does not match the active task Dockerfile"
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
    stage_model_config: Path | None = None,
    num_workers: int = 1,
    attempts: int = 1,
    max_turns: int | None = None,
    pier_executable: str = "pier",
    environment: str = "docker",
    dry_run: bool = False,
    p0_manifest: Path | None = None,
    fork_plan: Path | None = None,
    fork_plans: Path | None = None,
    workflow_timeout_sec: int | None = None,
    few_shot_dir: Path | None = None,
    review_arm: str = "E0",
    review_contract_version: str | None = None,
    review_prompt_version: str | None = None,
    task_ids: tuple[str, ...] | None = None,
    trial_cpus: int | None = None,
    trial_memory_mb: int | None = None,
) -> Path:
    if agent_id not in DEEPSWE_AGENTS:
        raise ValueError(f"Unsupported agent: {agent_id}")
    if agent_id in {
        "pi-review-repair",
        "pi-v5-review-repair",
        "pi-v6-review-repair",
        "pi-v62-review-repair",
    } and attempts != 1:
        raise ValueError("The review workflow forbids automatic task-level retries")
    if trial_cpus is not None and trial_cpus <= 0:
        raise ValueError("trial_cpus must be positive when provided")
    if trial_memory_mb is not None and trial_memory_mb <= 0:
        raise ValueError("trial_memory_mb must be positive when provided")
    stage_model_payload = None
    if stage_model_config is not None:
        stage_model_path = stage_model_config.expanduser().resolve()
        try:
            stage_model_payload = json.loads(stage_model_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Stage model config does not exist: {stage_model_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Stage model config is not valid JSON: {stage_model_path}") from error
    resolved_stage_models = PiStageModelConfig.from_value(
        stage_model_payload,
        fallback_model=model,
    )
    if stage_model_config is not None and agent_id not in {
        "pi",
        "pi-v5-review-repair",
        "pi-v6-review-repair",
        "pi-v62-review-repair",
    }:
        raise ValueError("--stage-model-config requires a Pi agent")
    if not dry_run:
        missing_keys = [
            key for key in resolved_stage_models.required_env_keys()
            if not os.environ.get(key)
        ]
        if missing_keys:
            raise RuntimeError(
                "Required provider API key(s) are not set: "
                + ", ".join(missing_keys)
            )
    resolved_checkout = validate_manifest_against_checkout(manifest, checkout)
    reviewer_manifest = (
        load_reviewer_experiment_manifest(p0_manifest) if p0_manifest is not None else None
    )
    if reviewer_manifest is not None:
        if agent_id not in {"pi-review-repair", "pi-v5-review-repair"}:
            raise ValueError("--p0-manifest requires a reviewer agent")
        if (
            reviewer_manifest.source_manifest_sha256
            != hashlib.sha256(manifest.path.read_bytes()).hexdigest()
        ):
            raise ValueError("P0 source manifest does not match the active DeepSWE manifest")
        if not set(reviewer_manifest.task_ids).issubset(manifest.task_ids):
            raise ValueError("Reviewer tasks are not a subset of the active DeepSWE manifest")
    available_task_ids = (
        reviewer_manifest.task_ids if reviewer_manifest is not None else manifest.task_ids
    )
    if fork_plan is not None and fork_plans is not None:
        raise ValueError("Use either --fork-plan or --fork-plans")
    resolved_fork_plans = V6ForkPlans.load(fork_plans) if fork_plans is not None else None
    if task_ids is not None:
        selected_task_ids = task_ids
    elif resolved_fork_plans is not None:
        selected_task_ids = tuple(resolved_fork_plans.plans)
    else:
        selected_task_ids = available_task_ids
    if len(selected_task_ids) != len(set(selected_task_ids)):
        raise ValueError("Selected DeepSWE task IDs must be unique")
    unknown = sorted(set(selected_task_ids) - set(available_task_ids))
    if not selected_task_ids or unknown:
        detail = ": " + ", ".join(unknown) if unknown else ""
        raise ValueError("Selected DeepSWE tasks are empty or outside the active manifest" + detail)
    resolved_fork_plan = V6ForkPlan.load(fork_plan) if fork_plan is not None else None
    if resolved_fork_plan is not None:
        if agent_id not in {"pi-v6-review-repair", "pi-v62-review-repair"}:
            raise ValueError("--fork-plan requires --agent pi-v6-review-repair or pi-v62-review-repair")
        if selected_task_ids != (resolved_fork_plan.task_id,):
            raise ValueError("--fork-plan requires exactly its task_id via --task-name")
        for source in resolved_fork_plan.sources.values():
            if not source.exists():
                raise ValueError(f"V6 fork source does not exist: {source}")
    if resolved_fork_plans is not None:
        if agent_id not in {"pi-v6-review-repair", "pi-v62-review-repair"}:
            raise ValueError("--fork-plans requires --agent pi-v6-review-repair or pi-v62-review-repair")
        if set(selected_task_ids) != set(resolved_fork_plans.plans):
            raise ValueError("--fork-plans task IDs must match the selected tasks")
        for plan in resolved_fork_plans.plans.values():
            for source in plan.sources.values():
                if not source.exists():
                    raise ValueError(f"V6 fork source does not exist: {source}")
    if reviewer_manifest is not None:
        _validate_reviewer_task_bases(
            checkout=resolved_checkout,
            manifest=reviewer_manifest,
            task_ids=selected_task_ids,
        )
    if agent_id == "pi-review-repair":
        contract_version = resolve_review_contract_version(
            review_arm, review_contract_version
        )
        prompt_version = resolve_review_prompt_version(
            contract_version, review_arm, review_prompt_version
        )
    else:
        contract_version = None
        prompt_version = None
        if review_arm != "E0":
            raise ValueError("--review-arm requires --agent pi-review-repair")
        if review_contract_version is not None:
            raise ValueError(
                "--review-contract-version requires --agent pi-review-repair"
            )
        if review_prompt_version is not None:
            raise ValueError("--review-prompt-version requires --agent pi-review-repair")
    frozen_few_shots = None
    few_shot_source_dir = None
    if agent_id == "pi-review-repair":
        assert contract_version is not None
        if few_shot_dir is None:
            raise ValueError("The review workflow requires --few-shot-dir")
        frozen_few_shots = load_frozen_few_shots(few_shot_dir)
        if frozen_few_shots.contract_version != contract_version:
            raise ValueError(
                f"Few-shot contract {frozen_few_shots.contract_version} is incompatible "
                f"with selected contract {contract_version}"
            )
        if frozen_few_shots.prompt_version != review_prompt_version:
            raise ValueError(
                "Few-shot prompt version does not match the selected reviewer prompt version"
            )
        few_shot_source_dir = frozen_few_shots.root
    elif few_shot_dir is not None:
        raise ValueError("--few-shot-dir is valid only with --agent pi-review-repair")
    executable = _resolve_pier_executable(pier_executable)
    if executable is None and not dry_run:
        raise RuntimeError(
            "Pier is not installed. Use Python 3.12+ and install the 'deepswe' extra."
        )
    executable = executable or pier_executable
    resolved_jobs_dir = jobs_dir.expanduser().resolve()
    if reviewer_manifest is None:
        default_name = default_job_name(agent_id, manifest)
    elif agent_id == "pi-v5-review-repair":
        default_name = (
            f"deepswe-v1.1-pi-v5-review-repair-random{len(reviewer_manifest.task_ids)}"
        )
    elif reviewer_manifest.raw.get("selection_method") == (
        "source-manifest-all-tasks-with-development-exclusions-v1"
    ):
        default_name = "deepswe-v1.1-pi-review-repair-random60"
    else:
        default_name = "deepswe-v1.1-pi-review-repair-nearmiss16"
    resolved_name = job_name or default_name
    job_dir = resolved_jobs_dir / resolved_name
    run_record = resolved_jobs_dir / f"{resolved_name}.run.json"
    if not dry_run:
        resolved_jobs_dir.mkdir(parents=True, exist_ok=True)
        if job_dir.exists():
            raise ValueError(
                f"Pier job directory already exists: {job_dir}. Choose a new --job-name "
                "or use Pier's resume command directly."
            )
        if run_record.exists():
            raise ValueError(
                f"Pier run record already exists: {run_record}. Choose a new --job-name."
            )
        if frozen_few_shots is not None:
            frozen_few_shots = freeze_few_shots(
                frozen_few_shots,
                resolved_jobs_dir / ".review_inputs" / frozen_few_shots.digest,
            )
    few_shot_dir = frozen_few_shots.root if frozen_few_shots is not None else None
    command = build_pier_command(
        pier_executable=executable,
        checkout=resolved_checkout,
        manifest=manifest,
        agent_id=agent_id,
        jobs_dir=resolved_jobs_dir,
        job_name=resolved_name,
        model=model,
        stage_model_config=stage_model_payload,
        num_workers=num_workers,
        attempts=attempts,
        max_turns=max_turns,
        environment=environment,
        reviewer_manifest=reviewer_manifest,
        fork_plan=(resolved_fork_plan.path if resolved_fork_plan is not None else None),
        fork_plans=(resolved_fork_plans.path if resolved_fork_plans is not None else None),
        workflow_timeout_sec=workflow_timeout_sec,
        few_shot_dir=few_shot_dir,
        few_shot_digest=(frozen_few_shots.digest if frozen_few_shots is not None else None),
        review_arm=review_arm,
        review_contract_version=contract_version,
        review_prompt_version=review_prompt_version,
        include_task_ids=selected_task_ids,
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
        "stage_models": resolved_stage_models.to_metadata(),
        "stage_model_config": str(stage_model_config.expanduser().resolve()) if stage_model_config else None,
        "fork_plan": str(resolved_fork_plan.path) if resolved_fork_plan else None,
        "fork_plan_sha256": resolved_fork_plan.digest if resolved_fork_plan else None,
        "fork_plans": str(resolved_fork_plans.path) if resolved_fork_plans else None,
        "fork_plans_sha256": resolved_fork_plans.digest if resolved_fork_plans else None,
        "fork_plan_sha256_by_task": (
            {task_id: plan.digest for task_id, plan in resolved_fork_plans.plans.items()}
            if resolved_fork_plans else None
        ),
        "environment": environment,
        "max_turns": max_turns if agent_id == "hermes" else None,
        "turn_limit_policy": (
            "agent-max-turns"
            if agent_id == "hermes" and max_turns is not None
            else "task-timeout-only"
        ),
        "manifest": str(manifest.path),
        "task_ids": list(selected_task_ids),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pier_cleanup_compat": environment == "docker",
        "docker_builder": builder_name,
        "docker_builder_cache_policy": (
            "retain-through-job-remove-after-job" if builder_name else None
        ),
        "docker_egress_proxy_prewarm": (
            {"status": "planned", "max_attempts": 3} if builder_name else None
        ),
        "command": command,
        "resource_isolation": {
            "trial_cpus": trial_cpus,
            "trial_memory_mb": trial_memory_mb,
            "concurrency": num_workers,
        },
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
        upstream_agent = (
            "pi"
            if agent_id
            in {
                "pi-review-repair",
                "pi-v5-review-repair",
                "pi-v6-review-repair",
                "pi-v62-review-repair",
            }
            else agent_id
        )
        metadata["agent_upstream_commit"] = AGENTS[upstream_agent].upstream_commit
        if agent_id == "pi-review-repair":
            metadata.update(
                {
                    "workflow": (f"solve-review-repair-once-v{contract_version.split('.')[0]}"),
                    "experiment_arm": review_arm,
                    "contract_version": contract_version,
                    "prompt_version": prompt_version,
                    "p0_manifest": (
                        str(reviewer_manifest.path) if reviewer_manifest is not None else None
                    ),
                    "source_job_id": (
                        reviewer_manifest.source_job_id if reviewer_manifest is not None else None
                    ),
                    "metric_task_ids": (
                        list(reviewer_manifest.metric_task_ids or reviewer_manifest.task_ids)
                        if reviewer_manifest is not None
                        else None
                    ),
                    "excluded_task_ids": (
                        list(reviewer_manifest.excluded_task_ids)
                        if reviewer_manifest is not None
                        else None
                    ),
                    "workflow_timeout_sec": workflow_timeout_sec,
                    "agent_timeout_multiplier": REVIEW_AGENT_TIMEOUT_MULTIPLIER,
                    "internal_role_timeouts_sec": {
                        **REVIEW_ROLE_TIMEOUTS_SEC,
                        **(
                            {
                                "solver": workflow_timeout_sec,
                                "repair": workflow_timeout_sec,
                            }
                            if workflow_timeout_sec is not None
                            else {}
                        ),
                    },
                    "few_shot_dir": str(few_shot_dir),
                    "few_shot_source_dir": str(few_shot_source_dir),
                    "few_shot_mode": frozen_few_shots.mode,
                    "few_shot_digest": frozen_few_shots.digest,
                    "few_shot_file_sha256": dict(frozen_few_shots.file_sha256),
                }
            )
        elif agent_id == "pi-v5-review-repair":
            metadata.update(
                {
                    "workflow": "solve-review-repair-v5",
                    "prompt_version": V5_PROMPT_VERSION,
                    "controller_version": V5_CONTROLLER_VERSION,
                    "p0_manifest": str(reviewer_manifest.path) if reviewer_manifest is not None else None,
                }
            )
        elif agent_id in {"pi-v6-review-repair", "pi-v62-review-repair"}:
            workflow_version = "6.2" if agent_id == "pi-v62-review-repair" else "6.1"
            metadata.update(
                {
                    "workflow": f"solve-review-repair-v{workflow_version}",
                    "prompt_version": "v6.2" if workflow_version == "6.2" else V6_PROMPT_VERSION,
                    "controller_version": "v6.2" if workflow_version == "6.2" else V6_CONTROLLER_VERSION,
                    "review_protocol": "v5",
                    "review_prompt_version": V5_PROMPT_VERSION,
                    "review_controller_version": V5_CONTROLLER_VERSION,
                    "ledger_timing": "before-solver",
                    "p0_manifest": None,
                    "fork_start_at": resolved_fork_plan.start_at if resolved_fork_plan else None,
                    "fork_start_at_by_task": (
                        {task_id: plan.start_at for task_id, plan in resolved_fork_plans.plans.items()}
                        if resolved_fork_plans else None
                    ),
                }
            )
    if dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return job_dir

    process_env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[3]
    python_path = process_env.get("PYTHONPATH")
    process_env["PYTHONPATH"] = (
        str(project_root) if not python_path else f"{project_root}{os.pathsep}{python_path}"
    )
    if builder_name:
        process_env["BUILDX_BUILDER"] = builder_name
        process_env[DEEPSWE_BUILDER_ENV] = builder_name
    process_env["EXTERNAL_EVAL_CONCURRENCY"] = str(num_workers)
    if trial_cpus is not None:
        process_env["EXTERNAL_EVAL_TRIAL_CPUS"] = str(trial_cpus)
    if trial_memory_mb is not None:
        process_env["EXTERNAL_EVAL_TRIAL_MEMORY_MB"] = str(trial_memory_mb)

    builder_created = False
    try:
        if builder_name:
            _create_dedicated_builder(builder_name)
            builder_created = True
            try:
                prewarm_attempts = _prewarm_egress_proxy(builder_name)
            except EgressProxyPrewarmError as error:
                metadata["docker_egress_proxy_prewarm"] = {
                    "status": (
                        "infra_failed_retryable"
                        if error.retryable
                        else "infra_failed_terminal"
                    ),
                    "attempts": error.attempts,
                    "max_attempts": 3,
                    "failure_class": "egress_proxy_prewarm",
                }
                atomic_write_json(run_record, metadata)
                raise
            metadata["docker_egress_proxy_prewarm"] = {
                "status": "completed",
                "attempts": prewarm_attempts,
                "max_attempts": 3,
            }
        atomic_write_json(run_record, metadata)
        _run_pier_process(command, process_env)
    finally:
        if builder_created and builder_name:
            _remove_dedicated_builder(builder_name)
    return job_dir
