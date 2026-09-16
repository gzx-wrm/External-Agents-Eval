from __future__ import annotations

import asyncio
import json
import os
import re
import resource
import shutil
import time
from pathlib import Path
from typing import Any

DEEPSWE_BUILDER_ENV = "EXTERNAL_AGENTS_EVAL_DEEPSWE_BUILDER"
_REGISTRY_CONTEXT_MARKERS = (
    "registry",
    "docker.io",
    "auth.docker",
    ".ecr.",
    "ghcr.io",
    "quay.io",
)


def install_docker_builder_patch(docker_environment_class: type[Any]) -> None:
    """Make Pier's compose builds explicitly use the job-scoped builder."""
    if getattr(docker_environment_class, "_external_eval_builder_patch", False):
        return

    original_build_command = docker_environment_class._build_command

    def build_command(*, force_build: bool) -> list[str]:
        command = original_build_command(force_build=force_build)
        builder_name = os.environ.get(DEEPSWE_BUILDER_ENV)
        if builder_name:
            command.extend(["--builder", builder_name])
        return command

    docker_environment_class._build_command = staticmethod(build_command)
    docker_environment_class._external_eval_builder_patch = True


def install_docker_image_cleanup_patch(trial_class: type[Any]) -> None:
    """Allow Pier's final cleanup to remove images after separate verification.

    Pier 0.3.1 stops the agent environment with ``keep_images=True`` before a
    separate verifier starts. That stop marks the environment as fully stopped,
    so the final ``delete=True`` stop is skipped and per-trial images accumulate.
    Keep the first stop, but leave the lifecycle open for the final image-removal
    pass.
    """
    if getattr(trial_class, "_external_eval_cleanup_patch", False):
        return

    original_stop = trial_class._stop_agent_environment

    async def stop_agent_environment(
        self: Any,
        *,
        keep_images: bool = False,
    ) -> None:
        await original_stop(self, keep_images=keep_images)
        if keep_images:
            self._is_agent_environment_stopped = False
            return

    trial_class._stop_agent_environment = stop_agent_environment
    trial_class._external_eval_cleanup_patch = True


def _is_transient_environment_start_error(error: BaseException | str) -> bool:
    """Recognize a small set of retryable build transports before agent startup."""
    messages: list[str] = []
    current: BaseException | None = error if isinstance(error, BaseException) else None
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    if not messages:
        messages.append(str(error))
    message = "\n".join(messages).lower()
    if any(
        marker in message
        for marker in (
            "environment start timed out",
            "environment build timed out",
            "docker daemon is not responding",
            "toomanyrequests",
            "rate limit",
        )
    ):
        return True
    if re.search(r"failed to fetch (?:anonymous|oauth) token", message):
        return True
    has_registry_context = any(marker in message for marker in _REGISTRY_CONTEXT_MARKERS)
    if has_registry_context and any(
        marker in message
        for marker in (
            "tls handshake timeout",
            "connection reset",
            "connection refused",
            "context canceled",
            "i/o timeout",
        )
    ):
        return True
    if bool(
        has_registry_context
        and "token" in message
        and re.search(r"\b(?:unexpected\s+)?eof\b", message)
    ):
        return True
    return (
        "pier-egress-proxy" in message
        and any(
        marker in message
        for marker in (
            "unable to connect",
            "could not connect",
            "connection failed",
            "connection timed out",
            "remote end closed connection",
            "temporary failure resolving",
        )
        )
    ) or any(
        marker in message
        for marker in (
            "error pulling image",
            "failed to solve",
            "failed to copy",
            "unexpected eof",
        )
    ) and any(marker in message for marker in ("docker", "buildkit", "registry", "image"))


def _write_start_record(
    execution: Any,
    *,
    status: str,
    attempts: int,
    retryable: bool | None = None,
) -> None:
    trial_paths = getattr(execution.environment, "trial_paths", None)
    trial_dir = getattr(trial_paths, "trial_dir", None)
    if trial_dir is None:
        return
    payload = {
        "schema_version": "1.0",
        "status": status,
        "attempts": attempts,
        "max_attempts": 3,
    }
    if retryable is not None:
        payload["retryable"] = retryable
        payload["failure_class"] = "environment_start"
    path = Path(trial_dir) / "environment_start.json"
    staging = path.with_suffix(".json.tmp")
    try:
        staging.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        staging.replace(path)
    except OSError:
        staging.unlink(missing_ok=True)
        return


def _write_start_attempts(execution: Any, attempts: list[dict[str, Any]]) -> None:
    trial_paths = getattr(execution.environment, "trial_paths", None)
    trial_dir = getattr(trial_paths, "trial_dir", None)
    if trial_dir is None:
        return
    path = Path(trial_dir) / "environment_start_attempts.json"
    staging = path.with_suffix(".json.tmp")
    try:
        staging.write_text(
            json.dumps(
                {"schema_version": "v1", "attempts": attempts},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(path)
    except OSError:
        staging.unlink(missing_ok=True)
        return


def install_environment_start_retry_patch(trial_execution_class: type[Any]) -> None:
    """Retry only recognized environment transport failures before an agent runs."""
    if getattr(trial_execution_class, "_external_eval_environment_retry_patch", False):
        return

    original_start = trial_execution_class.start_environment

    async def start_environment(self: Any, *, force_build: bool) -> None:
        max_attempts = 3
        history: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            try:
                await original_start(self, force_build=force_build)
                history.append({"attempt": attempt, "status": "succeeded"})
                _write_start_record(self, status="succeeded", attempts=attempt)
                _write_start_attempts(self, history)
                return
            except Exception as error:
                retryable = _is_transient_environment_start_error(error)
                history.append(
                    {
                        "attempt": attempt,
                        "status": "failed" if (not retryable or attempt == max_attempts) else "retrying",
                        "retryable": retryable,
                        "failure_class": "environment_start",
                        "error_type": type(error).__name__,
                        "error": str(error)[-4000:],
                    }
                )
                if not retryable or attempt == max_attempts:
                    _write_start_record(
                        self,
                        status=(
                            "infra_failed_retryable"
                            if retryable
                            else "infra_failed_terminal"
                        ),
                        attempts=attempt,
                        retryable=retryable,
                    )
                    _write_start_attempts(self, history)
                    raise
                _write_start_record(
                    self,
                    status="retrying",
                    attempts=attempt,
                    retryable=True,
                )
                self.environment.logger.warning(
                    "Transient environment failure before agent startup; retrying "
                    "environment start (%s/%s): %s",
                    attempt,
                    max_attempts,
                    error,
                )
                await asyncio.sleep(2 ** (attempt - 1))

    trial_execution_class.start_environment = start_environment
    trial_execution_class._external_eval_environment_retry_patch = True


def install_registry_start_retry_patch(trial_execution_class: type[Any]) -> None:
    """Backward-compatible name for the environment-start retry patch."""
    install_environment_start_retry_patch(trial_execution_class)


def classify_verifier_failure(error: BaseException | str | None) -> str:
    """Classify verifier failures without conflating them with test failures."""
    if error is None:
        return "none"
    messages: list[str] = []
    current: BaseException | None = error if isinstance(error, BaseException) else None
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    if not messages:
        messages.append(str(error))
    message = "\n".join(messages).lower()
    if "timeout" in message or "timed out" in message:
        return "verifier_timeout"
    if any(marker in message for marker in (
        "docker build", "failed to build", "image not found", "pull access",
        "manifest unknown", "buildkit", "failed to fetch",
    )):
        return "verifier_build_failure"
    if any(marker in message for marker in (
        "connection reset", "connection refused", "network is unreachable",
        "browser", "playwright", "broken pipe", "eof", "process exited",
        "transport",
    )):
        return "verifier_infrastructure_flake"
    if any(marker in message for marker in (
        "assertionerror", "assertion failed", "test failed", "failed:",
        "failures:", "no tests passed",
    )):
        return "test_failure"
    return "verifier_error"


def _resource_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    snapshot: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "process_user_cpu_sec": usage.ru_utime,
        "process_system_cpu_sec": usage.ru_stime,
        "max_rss": usage.ru_maxrss,
    }
    try:
        snapshot["disk_free_bytes"] = shutil.disk_usage(Path.cwd()).free
    except OSError:
        snapshot["disk_free_bytes"] = None
    return snapshot


def _trial_sidecar_path(trial: Any, name: str) -> Path | None:
    trial_paths = getattr(trial, "trial_paths", None) or getattr(trial, "_trial_paths", None)
    trial_dir = getattr(trial_paths, "trial_dir", None)
    return Path(trial_dir) / name if trial_dir is not None else None


def _atomic_sidecar(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def install_verifier_observability_patch(trial_class: type[Any]) -> None:
    """Persist verifier attempts and classify infrastructure jitter.

    Pier retries timeout exceptions internally.  Wrapping ``_verify_once``
    records each original attempt so a later successful retry cannot erase a
    timeout, process, browser, or verifier-build failure.
    """
    if getattr(trial_class, "_external_eval_verifier_observability_patch", False):
        return
    original_once = trial_class._verify_once

    async def verify_once(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        started_at = time.time()
        path = _trial_sidecar_path(self, "verifier_attempts.json")
        attempts: list[dict[str, Any]] = []
        if path is not None and path.is_file():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(prior, list):
                    attempts = prior
            except (OSError, json.JSONDecodeError):
                attempts = []
        attempt_index = len(attempts) + 1
        entry: dict[str, Any] = {
            "attempt": attempt_index,
            "started_at": started_at,
            "resource_before": _resource_snapshot(),
        }
        try:
            result = await original_once(self, *args, **kwargs)
        except asyncio.CancelledError as error:
            # ``asyncio.wait_for`` enforces the verifier timeout by cancelling
            # this coroutine.  CancelledError is a BaseException on supported
            # Python versions, so a normal ``except Exception`` would leave a
            # failed attempt without a class and make verifier jitter look like
            # an unrecorded transport failure.
            entry.update(
                {
                    "status": "failed",
                    "failure_class": "verifier_timeout",
                    "error_type": type(error).__name__,
                    "error": str(error)[-4000:],
                }
            )
            raise
        except Exception as error:
            entry.update(
                {
                    "status": "failed",
                    "failure_class": classify_verifier_failure(error),
                    "error_type": type(error).__name__,
                    "error": str(error)[-4000:],
                }
            )
            raise
        else:
            rewards = getattr(result, "rewards", None)
            entry.update(
                {
                    "status": "completed",
                    "failure_class": None,
                    "rewards": rewards,
                }
            )
            return result
        finally:
            entry.update(
                {
                    "finished_at": time.time(),
                    "wall_time_sec": time.monotonic() - started,
                    "resource_after": _resource_snapshot(),
                }
            )
            attempts.append(entry)
            if path is not None:
                _atomic_sidecar(path, attempts)
                summary_path = _trial_sidecar_path(self, "verifier_observability.json")
                if summary_path is not None:
                    failures = [item for item in attempts if item.get("status") == "failed"]
                    _atomic_sidecar(
                        summary_path,
                        {
                            "schema_version": "v1",
                            "attempt_count": len(attempts),
                            "retry_count": max(len(attempts) - 1, 0),
                            "original_failure_class": (
                                failures[0].get("failure_class") if failures else None
                            ),
                            "last_failure_class": (
                                failures[-1].get("failure_class") if failures else None
                            ),
                            "attempts": attempts,
                        },
                    )

    trial_class._verify_once = verify_once
    trial_class._external_eval_verifier_observability_patch = True


def install_resource_isolation_patch(trial_class: type[Any]) -> None:
    """Apply optional per-trial CPU/memory limits and record the allocation.

    Pier already creates separate verifier containers.  This patch makes the
    allocation explicit when the launcher provides limits, preventing a
    high-concurrency run from silently depending on host-wide defaults.
    """
    if getattr(trial_class, "_external_eval_resource_isolation_patch", False):
        return
    original_init = trial_class.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        config = getattr(self, "config", None)
        trial_environment_config = getattr(config, "environment", None)
        live_environment = getattr(self, "_environment", None)
        if live_environment is None:
            live_environment = getattr(self, "environment", None)
        if trial_environment_config is None and live_environment is None:
            return
        def _positive_int(name: str) -> int | None:
            value = os.environ.get(name)
            if value is None or not value.strip():
                return None
            try:
                parsed = int(value)
            except ValueError as error:
                raise RuntimeError(f"{name} must be a positive integer") from error
            if parsed <= 0:
                raise RuntimeError(f"{name} must be a positive integer")
            return parsed
        cpus = _positive_int("EXTERNAL_EVAL_TRIAL_CPUS")
        memory_mb = _positive_int("EXTERNAL_EVAL_TRIAL_MEMORY_MB")
        configured_cpus = getattr(trial_environment_config, "override_cpus", None)
        configured_memory = getattr(trial_environment_config, "override_memory_mb", None)
        if cpus is None:
            cpus = configured_cpus
        if memory_mb is None:
            memory_mb = configured_memory
        # The environment has already been constructed by Trial.__init__, so
        # changing TrialEnvironmentConfig alone would affect only a later
        # separate verifier.  Keep both the live agent environment and the
        # Trial config bound to the same limits.  Pier's BaseEnvironment reads
        # ``_override_*`` during construction and then derives Docker Compose
        # limits from ``task_env_config``.
        if cpus is not None:
            if live_environment is not None:
                live_environment._override_cpus = cpus
                live_task_config = getattr(live_environment, "task_env_config", None)
                if live_task_config is not None:
                    live_task_config.cpus = cpus
            if trial_environment_config is not None:
                trial_environment_config.override_cpus = cpus
        if memory_mb is not None:
            if live_environment is not None:
                live_environment._override_memory_mb = memory_mb
                live_task_config = getattr(live_environment, "task_env_config", None)
                if live_task_config is not None:
                    live_task_config.memory_mb = memory_mb
            if trial_environment_config is not None:
                trial_environment_config.override_memory_mb = memory_mb
        path = _trial_sidecar_path(self, "resource_limits.json")
        if path is not None:
            _atomic_sidecar(
                path,
                {
                    "schema_version": "v1",
                    "scope": "trial_and_separate_verifier",
                    "mode": (
                        "separate_container_with_explicit_limits"
                        if cpus is not None or memory_mb is not None
                        else "separate_container"
                    ),
                    "cpus": cpus,
                    "memory_mb": memory_mb,
                    "concurrency_hint": os.environ.get("EXTERNAL_EVAL_CONCURRENCY"),
                },
            )

    trial_class.__init__ = init
    trial_class._external_eval_resource_isolation_patch = True


def main() -> None:
    from pier.cli.main import app
    from pier.environments.docker.docker import DockerEnvironment
    from pier.trial.execution import TrialExecution
    from pier.trial.trial import Trial

    install_docker_builder_patch(DockerEnvironment)
    install_docker_image_cleanup_patch(Trial)
    install_environment_start_retry_patch(TrialExecution)
    install_verifier_observability_patch(Trial)
    install_resource_isolation_patch(Trial)
    app()


if __name__ == "__main__":
    main()
