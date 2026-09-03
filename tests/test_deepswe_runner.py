import asyncio
import json
from pathlib import Path

from external_agents_eval.benchmarks.deepswe import runner
from external_agents_eval.benchmarks.deepswe.config import (
    CODEX_CONFIG_FILE,
    CODEX_VERSION,
)
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
    assert "--agent" not in command
    assert "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" in command
    assert command.count("--include-task-name") == 2
    assert command[-4:] == [
        "--include-task-name",
        "task-b",
        "--include-task-name",
        "task-a",
    ]
    assert "--sample-seed" not in command


def test_codex_command_uses_pier_native_agent_and_secret_safe_config(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="codex",
        jobs_dir=Path("/jobs"),
        job_name="fixed-codex",
    )

    agent_index = command.index("--agent")
    assert command[agent_index + 1] == "codex"
    assert "--agent-import-path" not in command
    assert f"version={CODEX_VERSION}" in command
    assert f"config_toml_file={CODEX_CONFIG_FILE.resolve()}" in command
    assert "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" in command

    config = CODEX_CONFIG_FILE.read_text(encoding="utf-8")
    assert 'web_search = "disabled"' in config
    assert 'base_url = "https://api.deepseek.com/"' in config
    assert 'wire_api = "responses"' in config
    assert 'env_key = "DEEPSEEK_API_KEY"' in config
    assert "experimental_bearer_token" not in config


def test_codex_dry_run_records_pier_and_provider_metadata(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="codex",
        jobs_dir=tmp_path / "jobs",
        environment="modal",
        dry_run=True,
    )

    metadata = json.loads(capsys.readouterr().out)
    assert metadata["agent"] == "codex"
    assert metadata["agent_version"] == CODEX_VERSION
    assert metadata["agent_implementation"] == "pier-native"
    assert metadata["model_provider"] == "deepseek"
    assert metadata["model_provider_base_url"] == "https://api.deepseek.com/"
    assert metadata["model_provider_wire_api"] == "responses"
    assert "agent_upstream_commit" not in metadata


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


def test_builder_inherits_container_reachable_docker_desktop_proxy(monkeypatch):
    docker_info = {
        "HttpProxy": "http.docker.internal:3128",
        "HttpsProxy": "http.docker.internal:3128",
        "NoProxy": "hubproxy.docker.internal",
    }

    def fake_run(command, **_kwargs):
        assert command == ["docker", "info", "--format", "{{json .}}"]
        return runner.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps(docker_info),
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._docker_desktop_proxy_driver_options() == [
        "--driver-opt",
        "env.http_proxy=http://http.docker.internal:3128",
        "--driver-opt",
        "env.https_proxy=http://http.docker.internal:3128",
        "--driver-opt",
        "env.no_proxy=hubproxy.docker.internal",
    ]


def test_builder_does_not_forward_unreachable_or_credentialed_proxy(monkeypatch):
    docker_info = {
        "HttpProxy": "http://127.0.0.1:7890",
        "HttpsProxy": "http://user:secret@http.docker.internal:3128",
        "NoProxy": "localhost",
    }
    result = runner.subprocess.CompletedProcess(
        ["docker", "info"],
        returncode=0,
        stdout=json.dumps(docker_info),
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: result)

    assert runner._docker_desktop_proxy_driver_options() == []


def test_builder_creation_adds_detected_proxy_options(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return_code = 1 if command[:3] == ["docker", "buildx", "inspect"] else 0
        return runner.subprocess.CompletedProcess(command, returncode=return_code)

    proxy_options = [
        "--driver-opt",
        "env.https_proxy=http://http.docker.internal:3128",
    ]
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "_docker_desktop_proxy_driver_options",
        lambda: proxy_options,
    )

    runner._create_dedicated_builder("dedicated-builder")

    assert calls[-1] == [
        "docker",
        "buildx",
        "create",
        "--name",
        "dedicated-builder",
        "--driver",
        "docker-container",
        *proxy_options,
        "--bootstrap",
    ]


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
