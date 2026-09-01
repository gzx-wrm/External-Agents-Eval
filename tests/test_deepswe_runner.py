import asyncio
from pathlib import Path

from external_agents_eval.benchmarks.deepswe import runner
from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest
from external_agents_eval.benchmarks.deepswe.pier_compat import (
    DEEPSWE_BUILDER_ENV,
    install_docker_builder_patch,
    install_docker_image_cleanup_patch,
)
from external_agents_eval.benchmarks.deepswe.runner import (
    PIER_COMPAT_MODULE,
    _apply_docker_cleanup_compat,
    _docker_builder_name,
    _resolve_pier_executable,
    _run_pier_process,
    build_pier_command,
)


def _manifest(tmp_path):
    return DeepSWEManifest(
        path=tmp_path / "manifest.json",
        task_ids=("task-b", "task-a"),
        sample_seed=7,
        sample_size=2,
        population_count=113,
        population_fingerprint="a" * 64,
        raw={},
    )


def test_pier_command_uses_exact_manifest_and_secret_template(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi",
        jobs_dir=Path("/jobs"),
        job_name="fixed-pi",
    )
    assert "external_agents_eval.agents.pier.pi:PiPierAgent" in command
    assert "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" in command
    assert command.count("--include-task-name") == 2
    assert command[-4:] == [
        "--include-task-name",
        "task-b",
        "--include-task-name",
        "task-a",
    ]
    assert "--sample-seed" not in command


def test_hermes_command_carries_turn_limit(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="hermes",
        jobs_dir=Path("/jobs"),
        job_name="fixed-hermes",
        max_turns=12,
    )
    index = command.index("--agent-kwarg")
    assert command[index + 1] == "max_turns=12"


def test_pier_resolves_next_to_active_python_when_not_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.touch()
    pier = bin_dir / "pier"
    pier.touch(mode=0o755)
    monkeypatch.setattr(runner.sys, "executable", str(python))
    monkeypatch.setattr(runner.shutil, "which", lambda _command: None)

    assert _resolve_pier_executable("pier") == str(pier)


def test_docker_command_uses_cleanup_compat_wrapper():
    command = _apply_docker_cleanup_compat(["/venv/bin/pier", "run", "--yes"], "docker")
    assert command == [runner.sys.executable, "-m", PIER_COMPAT_MODULE, "run", "--yes"]
    assert _apply_docker_cleanup_compat(command, "modal") is command


def test_cleanup_patch_keeps_final_image_removal_pass():
    class FakeTrial:
        _external_eval_cleanup_patch = False

        def __init__(self):
            self._is_agent_environment_stopped = False
            self.calls = []

        async def _stop_agent_environment(self, *, keep_images=False):
            if self._is_agent_environment_stopped:
                return
            self.calls.append(keep_images)
            self._is_agent_environment_stopped = True

    install_docker_image_cleanup_patch(FakeTrial)
    trial = FakeTrial()
    asyncio.run(trial._stop_agent_environment(keep_images=True))
    assert trial._is_agent_environment_stopped is False
    asyncio.run(trial._stop_agent_environment())
    assert trial.calls == [True, False]
    assert trial._is_agent_environment_stopped is True


def test_builder_patch_adds_explicit_compose_builder(monkeypatch):
    class FakeDockerEnvironment:
        _external_eval_builder_patch = False

        @staticmethod
        def _build_command(*, force_build):
            return ["build", "--no-cache"] if force_build else ["build"]

    monkeypatch.setenv(DEEPSWE_BUILDER_ENV, "dedicated-builder")
    install_docker_builder_patch(FakeDockerEnvironment)

    assert FakeDockerEnvironment._build_command(force_build=False) == [
        "build",
        "--builder",
        "dedicated-builder",
    ]
    assert FakeDockerEnvironment._build_command(force_build=True) == [
        "build",
        "--no-cache",
        "--builder",
        "dedicated-builder",
    ]


def test_builder_name_is_stable_and_job_scoped(tmp_path):
    first = _docker_builder_name(tmp_path, "job-a")
    assert first == _docker_builder_name(tmp_path, "job-a")
    assert first != _docker_builder_name(tmp_path, "job-b")
    assert first.startswith("external-eval-deepswe-")


def test_pier_process_waits_for_cleanup_after_keyboard_interrupt(monkeypatch):
    class FakeProcess:
        def __init__(self):
            self.wait_calls = 0
            self.terminated = False

        def wait(self):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            return 130

        def terminate(self):
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)

    try:
        _run_pier_process(["pier", "run"], {})
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("KeyboardInterrupt should be re-raised after child cleanup")

    assert process.wait_calls == 2
    assert process.terminated is False
