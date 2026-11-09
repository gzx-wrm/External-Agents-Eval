from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.benchmarks.swebench.isolated_environment import task_container
from external_agents_eval.benchmarks.swebench.sanitize_repository import sanitize


def test_baseline_tree_survives_but_answer_commits_and_dangling_objects_do_not(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL).decode().strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.test")
    source = tmp_path / "source.txt"
    source.write_text("baseline\n")
    git("add", ".")
    git("commit", "-m", "public baseline")
    base = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    source.write_text("future answer\n")
    git("commit", "-am", "issue solution")
    answer = git("rev-parse", "HEAD")
    answer_blob = git("rev-parse", "HEAD:source.txt")
    git("remote", "add", "origin", "https://example.test/issue-answer.git")
    audit = sanitize(tmp_path, base)
    assert source.read_text() == "baseline\n"
    assert git("rev-parse", "HEAD^{tree}") == tree
    assert git("rev-list", "--all", "--count") == "1"
    assert not git("remote")
    assert audit["commit_object_count"] == 1
    for object_id in (base, answer, answer_blob):
        result = subprocess.run(["git", "-C", str(tmp_path), "cat-file", "-e", object_id], capture_output=True, check=False)
        assert result.returncode != 0
    assert not git("fsck", "--no-reflogs", "--unreachable")
    repeated = sanitize(tmp_path, audit["synthetic_commit"])
    assert repeated["synthetic_commit"] == audit["synthetic_commit"]


def test_task_container_has_no_network_mounts_or_credentials():
    captured = {}

    def create(*args, **kwargs):
        captured.update(kwargs)
        return "container"

    assert task_container(SimpleNamespace(containers=SimpleNamespace(create=create)), "image", name="test") == "container"
    assert captured["network_mode"] == "none"
    assert captured["cap_drop"] == ["ALL"]
    assert not captured.get("volumes")
    assert captured["privileged"] is False
    assert not any("KEY" in name or "TOKEN" in name for name in captured["environment"])


def test_mini_bash_read_is_visible_exactly_as_recorded():
    pytest.importorskip("minisweagent")
    from external_agents_eval.agents.mini_v62 import MiniBashEnvironment
    from external_agents_eval.workflows.solve_review_repair.ft_protocol import returned_span

    class Bridge:
        def exec_sync(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout="one\ntwo\nthree\n", stderr="")

    env = MiniBashEnvironment(Bridge(), "/base")
    output = env.execute({"command": "ft_read '{\"path\":\"/base/source.py\",\"limit\":2}'"})
    name, arguments, recorded = env.events[0]
    assert name == "read"
    assert recorded == output
    assert returned_span(output["output"], arguments, "one\ntwo\nthree\n") == (1, 2)


def test_mini_runner_uses_native_loop_and_reuses_reviewer_history(tmp_path, monkeypatch):
    pytest.importorskip("minisweagent")
    import minisweagent.models.litellm_model as models

    from external_agents_eval.agents.mini_v62 import MiniRoleRunner
    from external_agents_eval.workflows.solve_review_repair.models import RoleName

    seen = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.calls = 0

        def query(self, messages, **kwargs):
            seen.append(list(messages))
            self.calls += 1
            command = "submit_json '{}'"
            return {"role": "assistant", "content": "Return the stage result.",
                    "extra": {"actions": [{"command": command}], "cost": 0}}

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, *args):
            return [{"role": "user", "content": output["output"]} for output in outputs]

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    monkeypatch.setattr(models, "LitellmModel", FakeModel)

    class Bridge:
        def exec_sync(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{}\n", stderr="")

        async def upload_file(self, source, target):
            assert Path(source).exists()

    runner = MiniRoleRunner(environment=Bridge(), logs_root=tmp_path, model="deepseek-flash",
                            provider="deepseek", agent_version="2.4.6", env={}, redact=lambda text: text)
    for index in range(2):
        result = asyncio.run(runner.run_role(
            role=RoleName.V5_IMPLEMENTATION_REVIEWER, prompt=f"Review batch {index}", worktree="/base",
            tools=("read",), timeout_sec=30, artifact_prefix=f"review-{index}", session_id="reviewer-I",
        ))
        assert result.exit_code == 0
        assert json.loads(result.final_assistant_text) == {}
        native = json.loads(result.transcript_path.read_text())
        assert native["info"]["mini_version"] == "2.4.6"
    assert len(seen) == 2
    assert "Your assigned working directory is /base" in seen[0][0]["content"]
    assert any("Review batch 0" in message.get("content", "") for message in seen[1])
    assert any("Review batch 1" in message.get("content", "") for message in seen[1])


@pytest.mark.parametrize("command", [
    "ft_read '{\"path\":\"/base/source.py\",\"limit\":2}'",
    "cd /base && ft_read '{\"path\":\"source.py\",\"limit\":2}'",
    "cd /base && ft_read '{\"path\":\"source.py\",\"limit\":2}';",
])
def test_mini_cd_wrapped_reads_still_produce_instrumented_evidence(command):
    pytest.importorskip("minisweagent")
    from external_agents_eval.agents.mini_v62 import MiniBashEnvironment

    class Bridge:
        def exec_sync(self, command, **kwargs):
            assert command == "cat -- /base/source.py"
            return SimpleNamespace(return_code=0, stdout="one\ntwo\nthree\n", stderr="")

    env = MiniBashEnvironment(Bridge(), "/base", tools=("read", "grep", "find", "ls"))
    result = env.execute({"command": command})
    assert result["returncode"] == 0
    assert env.events[0][0] == "read"
    assert env.events[0][1]["path"] == "/base/source.py"
    assert env.events[0][2] == result


def test_search_uses_native_bash_and_records_the_actual_observation():
    pytest.importorskip("minisweagent")
    from external_agents_eval.agents.mini_v62 import MiniBashEnvironment

    class Bridge:
        def exec_sync(self, command, **kwargs):
            assert command == "cd /base && sed -n '1,40p' source.py"
            assert kwargs["cwd"] == "/base"
            return SimpleNamespace(return_code=0, stdout="source text\n", stderr="")

    env = MiniBashEnvironment(Bridge(), "/base", tools=("read", "grep", "find", "ls"))
    result = env.execute({"command": "cd /base && sed -n '1,40p' source.py"})
    assert result["returncode"] == 0
    assert result["output"] == "source text\n"
    assert env.events == [("bash", {"command": "cd /base && sed -n '1,40p' source.py",
                                   "cwd": "/base"}, result)]


def test_runtime_seed_copies_ignored_compiled_assets_without_changing_git_tree(tmp_path):
    from external_agents_eval.agents.mini_runtime_assets import seed, snapshot

    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.DEVNULL).decode().strip()

    git("init")
    (source / ".gitignore").write_text("*.so\n*.egg-info/\n_version.py\n*.cache\n")
    (source / "module.py").write_text("public baseline\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-m", "baseline")
    (source / "module.so").write_bytes(b"compiled-runtime")
    (source / "_version.py").write_text("version = '1.0'\n")
    (source / "other.cache").write_text("not a runtime asset")
    worktree = tmp_path / "worktree"
    git("worktree", "add", "--detach", str(worktree))
    report = snapshot(str(source), str(tmp_path / "snapshot"))
    assert {item["path"] for item in report["files"]} == {"module.so", "_version.py"}
    seed(str(tmp_path / "snapshot"), str(worktree))
    assert (worktree / "module.so").read_bytes() == b"compiled-runtime"
    assert (worktree / "module.so").stat().st_ino != (source / "module.so").stat().st_ino
    assert not (worktree / "other.cache").exists()
    assert not subprocess.check_output(["git", "-C", str(worktree), "status", "--porcelain"]).strip()


def test_bash_source_snapshot_uses_git_blobs_including_export_ignored_files(tmp_path):
    from external_agents_eval.agents.mini_v62 import MiniWorkspaceManager

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)

    git("init")
    (repo / ".gitattributes").write_text("ignored.txt export-ignore\nsource.txt export-subst\n")
    (repo / "ignored.txt").write_text("tracked despite export-ignore\n")
    (repo / "source.txt").write_text("$Format:%H$\n")
    (repo / "symbols-⊗.txt").write_text("unicode filename\n")
    (repo / "alias.txt").symlink_to("source.txt")
    (repo / "binary").write_bytes(b"a\0b")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-m", "baseline")
    revision = git("rev-parse", "HEAD").decode().strip()
    (repo / "source.txt").write_text("working tree edit\n")

    class Environment:
        async def exec(self, command, **kwargs):
            if command.startswith("python -c "):
                command = shlex.quote(sys.executable) + command[len("python"):]
                command = "LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 " + command
            result = await asyncio.to_thread(subprocess.run, command, shell=True,
                                            cwd=kwargs.get("cwd"), capture_output=True,
                                            text=True, check=False)
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

        async def download_file(self, source, destination):
            shutil.copyfile(source, destination)

    workspace = MiniWorkspaceManager(environment=Environment(), transfer_dir=tmp_path / "transfer")
    workspace.INPUT_ROOT = str(tmp_path / "contracts")
    Path(workspace.INPUT_ROOT).mkdir()
    files = asyncio.run(workspace.read_frozen_text_files(worktree=str(repo), base_commit=revision))
    assert files["ignored.txt"] == b"tracked despite export-ignore\n"
    assert files["source.txt"] == files["alias.txt"] == b"$Format:%H$\n"
    assert files["symbols-⊗.txt"] == b"unicode filename\n"
    assert "binary" not in files
    assert not list((tmp_path / "transfer").glob("*.tar"))
    assert not list(Path(workspace.INPUT_ROOT).glob("*.tar"))


def test_executed_but_unreturned_batch_results_are_not_ft_evidence():
    from external_agents_eval.agents.mini_v62 import model_visible_events

    first = ("bash", {"command": "cat a.py", "mini_tool_call_id": "read-a"}, {"output": "a source"})
    second = ("bash", {"command": "cat b.py", "mini_tool_call_id": "read-b"}, {"output": "b source"})
    visible, omitted = model_visible_events([first, second], [
        {"role": "tool", "tool_call_id": "read-a", "content": "<output>a source</output>"},
    ])
    assert visible == [first]
    assert [item["tool_call_id"] for item in omitted] == ["read-b"]


def test_observation_formatting_cannot_silently_add_unseen_text_to_ft():
    from external_agents_eval.agents.mini_v62 import model_visible_events

    events = [("bash", {"command": "cat a.py", "mini_tool_call_id": "a"},
               {"output": "complete original source"})]
    with pytest.raises(ValueError, match="refuse unseen evidence"):
        model_visible_events(events, [{"role": "tool", "tool_call_id": "a", "content": "complete..."}])
