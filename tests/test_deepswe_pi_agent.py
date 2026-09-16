import asyncio
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents.pier.pi import (
    PI_BUILTIN_TOOLS,
    PiPierAgent,
    _build_pi_command,
)
from external_agents_eval.agents.pier.pi_role_runner import PiRoleRunner
from external_agents_eval.agents.pier.pi_solve_review_repair import (
    PiSolveReviewRepairPierAgent,
    issue_title_from_instruction,
    resolve_frozen_task_id,
)
from external_agents_eval.agents.pier.pi_v5_solve_review_repair import (
    PiV5SolveReviewRepairPierAgent,
    managed_delivery_from_instruction,
)
from external_agents_eval.workflows.solve_review_repair.config import WorkflowConfig
from external_agents_eval.workflows.solve_review_repair.models import (
    IndeterminateInvocationError,
    RoleName,
)


def test_pi_pier_enables_every_builtin_tool():
    assert PI_BUILTIN_TOOLS == (
        "read",
        "bash",
        "edit",
        "write",
        "grep",
        "find",
        "ls",
    )
    command = _build_pi_command(model="deepseek-test", prompt_path="/tmp/instruction.txt")
    assert "--tools read,bash,edit,write,grep,find,ls" in command


def test_pi_pier_reads_prompt_from_stdin():
    command = _build_pi_command(
        model="deepseek-test",
        prompt_path="/tmp/instruction with spaces.txt",
    )
    assert "-p < '/tmp/instruction with spaces.txt'" in command
    assert "$(cat" not in command


def test_pi_pier_loads_nvm_without_auto_using_project_version():
    agent = object.__new__(PiPierAgent)
    install_spec = agent.install_spec()
    install_script = "\n".join(step.run for step in install_spec.steps)
    assert "curl --retry 5 --retry-all-errors --retry-delay 2" in install_script
    assert '. "$NVM_DIR/nvm.sh" --no-use' in install_script


def test_pi_role_command_can_disable_tools_for_json_correction():
    command = _build_pi_command(
        model="deepseek-test",
        prompt_path="/tmp/correction.txt",
        tools=(),
    )
    assert "--no-tools" in command


def test_json_corrector_has_only_read_only_repository_tools():
    tools = WorkflowConfig().tools_for(RoleName.JSON_CORRECTOR)
    assert tools == ("read", "grep", "find", "ls")
    assert not {"bash", "edit", "write"}.intersection(tools)


def test_pi_role_runner_counts_public_test_commands():
    trajectory = {
        "steps": [
            {
                "tool_calls": [
                    {
                        "function_name": "bash",
                        "arguments": {"command": "pytest -q && go test ./..."},
                    },
                    {
                        "function_name": "read",
                        "arguments": {"path": "tests/test_client.py"},
                    },
                ]
            },
            {
                "tool_calls": [
                    {
                        "function_name": "bash",
                        "arguments": {"command": "npm test"},
                    }
                ]
            },
        ]
    }
    assert PiRoleRunner._test_command_count(trajectory) == 2


def test_review_agent_recovers_unique_pier_truncated_task_id():
    frozen = (
        "bandit-interprocedural-taint-checks",
        "claude-code-by-agents-recursive-delegation",
        "dynamodb-toolbox-lazy-recursive-schemas",
        "httpx-deterministic-cookie-store",
    )
    assert (
        resolve_frozen_task_id("bandit-interprocedural-taint-che", frozen)
        == "bandit-interprocedural-taint-checks"
    )
    assert (
        resolve_frozen_task_id("httpx-deterministic-cookie-store", frozen)
        == "httpx-deterministic-cookie-store"
    )
    assert (
        resolve_frozen_task_id("claude-code-by-agents-recursive", frozen)
        == "claude-code-by-agents-recursive-delegation"
    )
    assert (
        resolve_frozen_task_id("dynamodb-toolbox-lazy-recursive", frozen)
        == "dynamodb-toolbox-lazy-recursive-schemas"
    )


@pytest.mark.parametrize(
    ("session_task_id", "frozen", "message"),
    (
        ("unknown", ("unknown-but-not-a-pier-prefix",), "not in the frozen"),
        ("x" * 32, ("different-task",), "not in the frozen"),
        ("x" * 32, ("x" * 32 + "-a", "x" * 32 + "-b"), "ambiguous"),
    ),
)
def test_review_agent_rejects_untrusted_or_ambiguous_task_prefix(
    session_task_id,
    frozen,
    message,
):
    with pytest.raises(ValueError, match=message):
        resolve_frozen_task_id(session_task_id, frozen)


def test_review_agent_uses_heading_or_task_id_for_schema_bounded_issue_title():
    assert issue_title_from_instruction("# Short title\n\nBody", "task-a") == "Short title"
    assert issue_title_from_instruction("x" * 500, "task-a") == "x" * 500
    assert issue_title_from_instruction("x" * 501, "task-a") == "task-a"
    assert issue_title_from_instruction("\n\n", "task-a") == "task-a"


def test_review_agent_normalizes_numeric_prompt_version(monkeypatch):
    monkeypatch.setattr(PiPierAgent, "__init__", lambda self, *args, **kwargs: None)
    agent = PiSolveReviewRepairPierAgent(review_prompt_version=4.0)

    assert agent.review_prompt_version == "4.0"


def test_v5_review_agent_entrypoint_has_independent_configuration(monkeypatch):
    monkeypatch.setattr(PiPierAgent, "__init__", lambda self, *args, **kwargs: None)
    agent = PiV5SolveReviewRepairPierAgent(workflow_timeout_sec=3600)

    assert agent.name() == "pi-v5-review-repair"
    assert agent.workflow_timeout_sec == 3600
    assert agent.p0_manifest is None


def test_v5_adapter_resolves_abbreviated_manifest_commit_to_full_sha():
    full_sha = "7cef35b" + "a" * 33
    commands = []

    class Environment:
        async def exec(self, command, **kwargs):
            commands.append((command, kwargs))
            return SimpleNamespace(return_code=0, stdout=full_sha + "\n", stderr="")

    agent = object.__new__(PiV5SolveReviewRepairPierAgent)
    agent.workdir = "/app"
    agent.redact = lambda value: value

    resolved = asyncio.run(agent._resolve_commit(Environment(), "7cef35b"))

    assert resolved == full_sha
    assert commands == [
        ("git rev-parse --verify '7cef35b^{commit}'", {
            "cwd": "/app",
            "env": None,
            "user": None,
            "timeout_sec": None,
        })
    ]


def test_v5_adapter_rejects_missing_or_ambiguous_manifest_commit():
    class Environment:
        async def exec(self, _command, **_kwargs):
            return SimpleNamespace(return_code=128, stdout="", stderr="unknown revision")

    agent = object.__new__(PiV5SolveReviewRepairPierAgent)
    agent.workdir = "/app"
    agent.redact = lambda value: value

    with pytest.raises(RuntimeError, match="Command exited with status 128"):
        asyncio.run(agent._resolve_commit(Environment(), "7cef35b"))


def test_v5_adapter_binds_only_exact_deepswe_delivery_boilerplate():
    quote = (
        "Please work on this in a new branch from main and commit everything "
        "when you are done."
    )
    assert managed_delivery_from_instruction(f"Add behavior.\n\n{quote}") == (
        {"kind": "task", "field": "description", "quote": quote},
    )
    assert managed_delivery_from_instruction(
        "Build a product that creates a branch and commit."
    ) == ()


class _SessionEnvironment:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.pi_commands: list[str] = []
        self.settings_commands: list[str] = []

    @staticmethod
    def _result(code=0, stdout="", stderr=""):
        return SimpleNamespace(return_code=code, stdout=stdout, stderr=stderr)

    async def upload_file(self, source: Path, destination: str):
        self.files[destination] = source.read_bytes()

    async def download_file(self, source: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[source])

    async def exec(self, command: str, **kwargs):
        del kwargs
        if command.startswith("mkdir ") and not command.startswith("mkdir -p "):
            path = shlex.split(command)[1]
            if path in self.directories:
                return self._result(1)
            self.directories.add(path)
            return self._result()
        if command.startswith("if test -d ") and "-printf" in command:
            roots = re.findall(r"find (\S+) -type f", command)
            lines = []
            for root in roots:
                category = root.rsplit("/", 1)[-1]
                lines.extend(
                    f"{category}/{path.removeprefix(root + '/')}"
                    for path in self.files
                    if path.startswith(root + "/")
                )
            return self._result(stdout="\n".join(sorted(lines)) + ("\n" if lines else ""))
        if command.startswith("test -f ") and "wc -c <" in command:
            path = shlex.split(command)[-1]
            if path not in self.files:
                return self._result(1)
            return self._result(stdout=f"{len(self.files[path])}\n")
        if command.startswith("test -f "):
            path = shlex.split(command)[2]
            return self._result(0 if path in self.files else 1)
        if command.startswith("test -e "):
            path = shlex.split(command)[2]
            return self._result(0 if path in self.directories or path in self.files else 1)
        if command.startswith(("sha256sum ", "if test -f ")):
            candidates = [
                token.rstrip(";")
                for token in shlex.split(command)
                if token.rstrip(";") in self.files
            ]
            if not candidates:
                return self._result()
            path = candidates[-1]
            digest = hashlib.sha256(self.files[path]).hexdigest()
            return self._result(stdout=f"{digest}  {path}\n")
        if command.startswith("find ") and ".jsonl" in command:
            root = shlex.split(command)[1]
            matches = sorted(
                path for path in self.files if path.startswith(root + "/") and path.endswith(".jsonl")
            )
            return self._result(stdout=(matches[-1] + "\n") if matches else "")
        if command.startswith("cat "):
            path = shlex.split(command)[1]
            return self._result(stdout=self.files[path].decode())
        if command.startswith("git rev-parse --is-inside-work-tree"):
            return self._result(1)
        if "/opt/pi-runtime/bin/node" in command:
            self.pi_commands.append(command)
            transcript = re.search(r"> (\S+/pi\.jsonl)", command).group(1)
            stderr = re.search(r"2> (\S+/stderr\.txt)", command).group(1)
            events = [
                {"type": "session", "id": "review-session"},
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "model": "deepseek-test",
                        "content": [{"type": "text", "text": "{}"}],
                    },
                },
            ]
            self.files[transcript] = (
                "\n".join(json.dumps(event) for event in events) + "\n"
            ).encode()
            self.files[stderr] = b""
            session_match = re.search(
                r"--session-dir (\S+) --session-id (\S+)", command
            )
            if session_match:
                session_path = (
                    f"{session_match.group(1)}/"
                    f"2026-09-11T00-00-00-000Z_{session_match.group(2)}.jsonl"
                )
                self.files[session_path] = b'{"type":"session"}\n'
            else:
                exact_match = re.search(r"--session (\S+)", command)
                if exact_match:
                    self.files[exact_match.group(1)] += b'{"type":"message"}\n'
            completion = re.search(r"(/external_eval/v5_invocations/[^ ]+/completion\.json)", command)
            if completion:
                invocation = completion.group(1).split("/")[-2]
                self.files[completion.group(1)] = json.dumps(
                    {
                        "invocation_id": invocation,
                        "exit_code": 0,
                        "transcript_path": transcript,
                        "stderr_path": stderr,
                        "transcript_sha256": hashlib.sha256(
                            self.files[transcript]
                        ).hexdigest(),
                        "stderr_sha256": hashlib.sha256(
                            self.files[stderr]
                        ).hexdigest(),
                        "finished_at": "2026-09-11T00:00:01+00:00",
                    }
                ).encode()
            return self._result()
        if "/external_eval/v5_pi_settings" in command:
            self.settings_commands.append(command)
        for token in shlex.split(command.replace(";", " ")):
            if token.startswith("/") and "v5_invocations" in token:
                self.directories.add(token.rstrip("/"))
        return self._result()


def test_pi_role_runner_resolves_exact_session_file_and_disables_compaction(tmp_path):
    environment = _SessionEnvironment()
    runner = PiRoleRunner(
        environment=environment,
        logs_root=tmp_path,
        model="deepseek-test",
        agent_version="test",
        env={},
        redact=lambda value: value,
    )
    common = {
        "role": RoleName.V5_IMPLEMENTATION_REVIEWER,
        "worktree": "/candidate",
        "tools": ("read",),
        "timeout_sec": 30,
        "session_id": "review-session",
        "session_dir": "/external_eval/v5_sessions/I",
        "no_compaction": True,
    }
    first = asyncio.run(
        runner.run_role(
            **common,
            prompt="round one",
            artifact_prefix="v5/I/one",
            invocation_id="inv-1",
        )
    )
    # A restarted Controller constructs a fresh runner. It must rediscover the
    # one exact timestamped session file before sending the next round.
    runner = PiRoleRunner(
        environment=environment,
        logs_root=tmp_path,
        model="deepseek-test",
        agent_version="test",
        env={},
        redact=lambda value: value,
    )
    second = asyncio.run(
        runner.run_role(
            **common,
            prompt="round two",
            artifact_prefix="v5/I/two",
            invocation_id="inv-2",
        )
    )
    assert first.remote_session_file == second.remote_session_file
    assert first.pre_session_sha256 is None
    assert second.pre_session_sha256 == first.post_session_sha256
    assert "--session-dir /external_eval/v5_sessions/I --session-id review-session" in environment.pi_commands[0]
    assert f"--session {first.remote_session_file}" in environment.pi_commands[1]
    assert "--session-id" not in environment.pi_commands[1]
    assert all("--no-compaction" not in command for command in environment.pi_commands)
    assert all(
        subprocess.run(
            ["bash", "-n", "-c", command],
            check=False,
            capture_output=True,
        ).returncode
        == 0
        for command in environment.pi_commands
    )
    assert environment.settings_commands
    assert '"compaction":{"enabled":false}' in environment.settings_commands[0]


def test_pi_role_runner_replays_completed_invocation_without_second_process(tmp_path):
    environment = _SessionEnvironment()
    runner = PiRoleRunner(
        environment=environment,
        logs_root=tmp_path,
        model="deepseek-test",
        agent_version="test",
        env={},
        redact=lambda value: value,
    )
    common = {
        "role": RoleName.V5_SUMMARY_REVIEWER,
        "prompt": "summarize the frozen evidence",
        "worktree": "/candidate",
        "tools": ("read",),
        "timeout_sec": 30,
        "invocation_id": "same-invocation",
    }

    first = asyncio.run(
        runner.run_role(**common, artifact_prefix="v5/S/first")
    )
    replay = asyncio.run(
        runner.run_role(**common, artifact_prefix="v5/S/replay")
    )

    assert first.exit_code == replay.exit_code == 0
    assert len(environment.pi_commands) == 1
    assert replay.resource_usage["invocation_replayed"] is True

    transcript = "/external_eval/v5_invocations/same-invocation/pi.jsonl"
    environment.files[transcript] += b'{"type":"tampered"}\n'
    with pytest.raises(
        IndeterminateInvocationError, match="completion record does not match"
    ):
        asyncio.run(
            runner.run_role(**common, artifact_prefix="v5/S/tampered-replay")
        )

    with pytest.raises(
        IndeterminateInvocationError, match="different execution inputs"
    ):
        asyncio.run(
            runner.run_role(
                **{**common, "prompt": "different prompt"},
                artifact_prefix="v5/S/conflict",
            )
        )


def test_pi_role_runner_keeps_uncertain_remote_invocation_recoverable(tmp_path):
    class InterruptedEnvironment(_SessionEnvironment):
        async def exec(self, command: str, **kwargs):
            if "/opt/pi-runtime/bin/node" in command:
                raise TimeoutError("transport stopped waiting before completion")
            return await super().exec(command, **kwargs)

    runner = PiRoleRunner(
        environment=InterruptedEnvironment(),
        logs_root=tmp_path,
        model="deepseek-test",
        agent_version="test",
        env={},
        redact=lambda value: value,
    )

    with pytest.raises(IndeterminateInvocationError, match="no trusted completion"):
        asyncio.run(
            runner.run_role(
                role=RoleName.V5_IMPLEMENTATION_REVIEWER,
                prompt="inspect the candidate",
                worktree="/candidate",
                tools=("read",),
                timeout_sec=1,
                artifact_prefix="v5/I/interrupted",
                invocation_id="interrupted-invocation",
            )
        )


def test_pi_role_runner_archives_bounded_remote_review_materials(tmp_path):
    environment = _SessionEnvironment()
    output_root = "/tmp/external_eval/v5_review_notes/run/I/round-1/attempt-1"
    environment.directories.update(
        {output_root, f"{output_root}/probes", f"{output_root}/logs"}
    )
    environment.files[f"{output_root}/probes/repro.sh"] = b"#!/bin/sh\nexit 0\n"
    environment.files[f"{output_root}/logs/result.txt"] = b"exit=0\n"
    runner = PiRoleRunner(
        environment=environment,
        logs_root=tmp_path,
        model="deepseek-test",
        agent_version="test",
        env={},
        redact=lambda value: value,
    )

    files, manifest = asyncio.run(
        runner.download_review_materials(
            output_root=output_root,
            host_root=tmp_path / "downloaded",
        )
    )

    assert files == {
        "materials/logs/result.txt": b"exit=0\n",
        "materials/probes/repro.sh": b"#!/bin/sh\nexit 0\n",
    }
    assert {item["material_type"] for item in manifest} == {"log", "probe"}
    assert all(len(item["sha256"]) == 64 for item in manifest)


def test_pi_role_runner_allows_unrelated_scratch_symlink_but_rejects_note_escape(tmp_path):
    class FilesystemEnvironment:
        async def exec(self, command, **kwargs):
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=kwargs.get("cwd"),
                capture_output=True,
                text=True,
                check=False,
            )
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        async def download_file(self, source, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    runner = PiRoleRunner(
        environment=FilesystemEnvironment(),
        logs_root=tmp_path,
        model="deepseek-test",
        agent_version="test",
        env={},
        redact=lambda value: value,
    )
    key = f"symlink-{tmp_path.name}"
    output_root = asyncio.run(runner.prepare_review_output_root(key))
    root = Path(output_root)
    outside = root.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    try:
        (root / "node_modules").symlink_to(outside, target_is_directory=True)
        (root / "notes").mkdir()
        (root / "notes/O001.md").write_text("safe note\n", encoding="utf-8")
        assert asyncio.run(runner.resume_review_output_root(output_root)) == output_root
        assert asyncio.run(
            runner.download_review_file(
                output_root=output_root,
                relative_path="notes/O001.md",
                host_path=tmp_path / "downloaded.md",
            )
        ) == b"safe note\n"

        (root / "notes/O002.md").symlink_to(outside / "missing.md")
        with pytest.raises(RuntimeError, match="Unsafe or missing"):
            asyncio.run(
                runner.download_review_file(
                    output_root=output_root,
                    relative_path="notes/O002.md",
                    host_path=tmp_path / "escaped.md",
                )
            )
    finally:
        asyncio.run(runner.cleanup_review_output_root(output_root))
        shutil.rmtree(outside, ignore_errors=True)
