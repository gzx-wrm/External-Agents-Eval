from __future__ import annotations

import asyncio
import os
from typing import Any

DEEPSWE_BUILDER_ENV = "EXTERNAL_AGENTS_EVAL_DEEPSWE_BUILDER"
_builder_prune_lock = asyncio.Lock()


async def _prune_dedicated_builder(builder_name: str) -> None:
    """Remove unused cache only from this DeepSWE job's builder."""
    async with _builder_prune_lock:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "buildx",
            "prune",
            "--builder",
            builder_name,
            "--force",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            output = stdout.decode(errors="replace").strip() if stdout else ""
            raise RuntimeError(f"Failed to prune dedicated builder {builder_name!r}: {output}")


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

        builder_name = os.environ.get(DEEPSWE_BUILDER_ENV)
        if builder_name:
            try:
                await _prune_dedicated_builder(builder_name)
            except (OSError, RuntimeError) as error:
                self._logger.warning(
                    "Failed to prune DeepSWE builder %s after trial cleanup: %s",
                    builder_name,
                    error,
                )

    trial_class._stop_agent_environment = stop_agent_environment
    trial_class._external_eval_cleanup_patch = True


def main() -> None:
    from pier.cli.main import app
    from pier.environments.docker.docker import DockerEnvironment
    from pier.trial.trial import Trial

    install_docker_builder_patch(DockerEnvironment)
    install_docker_image_cleanup_patch(Trial)
    app()


if __name__ == "__main__":
    main()
