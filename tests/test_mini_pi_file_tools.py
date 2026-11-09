from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents import mini_v62
from external_agents_eval.agents import pi_file_tools as transport
from external_agents_eval.agents.pi_file_tools import (
    PI_EVIDENCE_PROTOCOL,
    PI_TOOL_NAMES,
    load_tool_schemas,
    runtime_directory,
)
from external_agents_eval.workflows.solve_review_repair.ft_evidence import observed_evidence
from external_agents_eval.workflows.solve_review_repair.ft_protocol import returned_span
from external_agents_eval.workflows.solve_review_repair.models import RoleName

HELPER = Path(mini_v62.__file__).with_name("pi_file_tools.mjs")


def test_other_host_binaries_cannot_be_uploaded_to_linux_task_containers(tmp_path, monkeypatch):
    (tmp_path / "runtime.tar.gz").write_bytes(b"reference-runtime")
    (tmp_path / "runtime.json").write_text(json.dumps({"platform": "Darwin", "machine": "arm64"}))
    monkeypatch.setattr(transport, "runtime_directory", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="Linux x86_64 host"):
        asyncio.run(transport.prepare_tools(None))


@pytest.fixture(scope="module")
def runtime():
    root = runtime_directory()
    if not (root / "bin/node").is_file():
        pytest.skip("Run prepare_pi_tools.py to provision the pinned reference tools")
    return root


def invoke(runtime, cwd, name, arguments):
    request = json.dumps({"name": name, "arguments": arguments, "cwd": str(cwd), "tool_call_id": "test"})
    return json.loads(subprocess.check_output([
        str(runtime / "bin/node"), str(HELPER), str(runtime), request,
    ], text=True, cwd=cwd))


@pytest.mark.parametrize("source,arguments,expected", [
    ("one\ntwo\nthree\n", {}, (1, 3)),
    ("one\ntwo\nthree\n", {"offset": 2, "limit": 1}, (2, 2)),
    ("one\ntwo\nthree", {"offset": 2, "limit": 20}, (2, 3)),
    ("\n=====\n{\n}\n\n", {"limit": 5}, (1, 5)),
    ("\n".join(str(i) for i in range(2003)), {}, (1, 2000)),
    (("汉" * 100 + "\n") * 300, {}, (1, 170)),
], ids=["whole-file", "offset-limit", "eof", "repeated-symbols", "line-cap", "utf8-byte-cap"])
def test_real_pi_read_returns_exactly_the_span_accepted_by_historical_ft(runtime, tmp_path, source, arguments, expected):
    (tmp_path / "source.txt").write_text(source)
    result = invoke(runtime, tmp_path, "read", {"path": "source.txt", **arguments})
    assert result["returncode"] == 0
    assert result["resolved_path"] == str(tmp_path / "source.txt")
    assert returned_span(result["output"], arguments, source) == expected


def test_real_pi_tools_keep_search_semantics_and_ignore_rules(runtime, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/api.py").write_text("before\nNeedle\nafter\n")
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "ignored.py").write_text("Needle\n")
    (tmp_path / ".hidden").write_text("hidden\n")
    grep = invoke(runtime, tmp_path, "grep", {"pattern": "needle", "ignoreCase": True,
                                           "context": 1})
    assert grep["returncode"] == 0
    assert grep["output"] == "src/api.py-1- before\nsrc/api.py:2: Needle\nsrc/api.py-3- after"
    filtered = invoke(runtime, tmp_path, "grep", {"pattern": "Needle", "glob": "src/*.py"})
    assert filtered["output"] == "src/api.py:2: Needle"
    find = invoke(runtime, tmp_path, "find", {"pattern": "src/**/*.py"})
    assert find["output"] == "src/api.py"
    listing = invoke(runtime, tmp_path, "ls", {})
    assert {"src/", ".hidden", ".gitignore"} <= set(listing["output"].splitlines())
    assert invoke(runtime, tmp_path, "grep", {"pattern": "missing"})["output"] == "No matches found"


def test_pi_read_errors_and_oversized_lines_do_not_become_source(runtime, tmp_path):
    (tmp_path / "big.txt").write_text("a" * (51 * 1024))
    large = invoke(runtime, tmp_path, "read", {"path": "big.txt"})
    assert large["returncode"] == 0
    assert large["source_text_returned"] is False
    missing = invoke(runtime, tmp_path, "read", {"path": "missing.txt"})
    assert missing["returncode"] == 1
    beyond = invoke(runtime, tmp_path, "read", {"path": "big.txt", "offset": 9})
    assert beyond["returncode"] == 1
    assert "beyond end of file" in beyond["output"]


def test_model_schemas_and_plain_completion_have_no_bash_or_finish(runtime):
    pytest.importorskip("minisweagent")
    from minisweagent.exceptions import FormatError

    schemas = load_tool_schemas(runtime)
    assert {x["function"]["name"] for x in schemas} == set(PI_TOOL_NAMES)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        tool_calls=[], content='{"summary":"done"}',
    ))])
    assert mini_v62.parse_pi_actions(response, schemas) == [{"final": '{"summary":"done"}'}]
    response.choices[0].message.tool_calls = [SimpleNamespace(
        id="bad", function=SimpleNamespace(name="bash", arguments='{"command":"cat source.py"}'),
    )]
    with pytest.raises(FormatError):
        mini_v62.parse_pi_actions(response, schemas)
    with pytest.raises(ValueError, match="Unknown Pi file tool"):
        mini_v62.MiniPiFileEnvironment(None, "/base").execute({"name": "bash", "arguments": {}})


def test_native_mini_loop_uses_pi_tools_and_delivers_historical_ft_evidence(runtime, tmp_path, monkeypatch):
    pytest.importorskip("minisweagent")
    import litellm

    base = tmp_path / "baseline"
    base.mkdir()
    source = "=====\n{\n}\nreturn_value\n"
    (base / "source.txt").write_text(source)
    received = []
    responses = [
        {"role": "assistant", "content": "Find the implementation.", "tool_calls": [{
            "id": "ls-1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "assistant", "content": "Read it.", "tool_calls": [{
            "id": "read-1", "type": "function", "function": {
                "name": "read", "arguments": '{"path":"source.txt","limit":3}'}}]},
        {"role": "assistant", "content": '{"summary":"Exploration draft."}'},
    ]

    def completion(**kwargs):
        received.append(kwargs)
        index = len(received) - 1
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "message": responses[index], "finish_reason": "tool_calls" if index < 2 else "stop",
        }], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    async def prepared(environment):
        return runtime

    def execute(bridge, cwd, name, arguments, tool_call_id):
        return invoke(runtime, cwd, name, arguments)

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(mini_v62, "prepare_tools", prepared)
    monkeypatch.setattr(mini_v62, "execute_tool", execute)

    class Bridge:
        async def upload_file(self, source, destination):
            assert source.is_file()

    runner = mini_v62.MiniRoleRunner(
        environment=Bridge(), logs_root=tmp_path / "logs", model="deepseek-flash",
        provider="deepseek", agent_version="2.4.6", env={}, redact=lambda text: text,
    )
    result = asyncio.run(runner.run_role(
        role=RoleName.V5_SPECIFICATION_REVIEWER, prompt="Inspect the repository, then return a JSON draft.",
        worktree=str(base), tools=PI_TOOL_NAMES, timeout_sec=30, artifact_prefix="A",
    ))
    assert result.exit_code == 0
    assert result.final_assistant_text == '{"summary":"Exploration draft."}'
    assert len(received) == 3
    assert all({x["function"]["name"] for x in call["tools"]} == set(PI_TOOL_NAMES) for call in received)
    tool_messages = [m for m in received[-1]["messages"] if m["role"] == "tool"]
    assert tool_messages[-1]["content"] == "=====\n{\n}\n\n[2 more lines in file. Use offset=4 to continue.]"
    trajectory = json.loads(result.trajectory_path.read_text())
    assert trajectory["evidence_protocol"] == PI_EVIDENCE_PROTOCOL
    assert [c["function_name"] for s in trajectory["steps"] for c in s.get("tool_calls", [])] == ["ls", "read"]

    class Workspace:
        async def read_frozen_file(self, *, path, **kwargs):
            return (path, source.encode())

    packet = asyncio.run(observed_evidence(
        trajectory, task_id="fixture", base_root=str(base), base_commit="frozen", workspace=Workspace(),
    ))
    assert [(e["path"], e["line_start"], e["line_end"], e["text"]) for e in packet["evidence"]] == [
        ("source.txt", 1, 3, "=====\n{\n}\n"),
    ]
    assert "bash_evidence_audit" not in packet
    assert "Exploration draft" not in json.dumps(packet)


def test_failed_read_notice_does_not_contaminate_another_valid_read():
    trajectory = {"evidence_protocol": PI_EVIDENCE_PROTOCOL, "steps": [{
        "step_id": 1,
        "tool_calls": [{"tool_call_id": "missing", "function_name": "read", "arguments": {"path": "missing"}},
                       {"tool_call_id": "ok", "function_name": "read", "arguments": {"path": "/base/a"}}],
        "observation": {"results": [{"source_call_id": "missing", "content": "ENOENT", "exit_code": 1},
                                    {"source_call_id": "ok", "content": "source", "exit_code": 0}]},
    }]}

    class Workspace:
        async def read_frozen_file(self, *, path, **kwargs):
            return path, b"source\n"

    packet = asyncio.run(observed_evidence(
        trajectory, task_id="fixture", base_root="/base", base_commit="frozen", workspace=Workspace(),
    ))
    assert packet["evidence"][0]["text"] == "source\n"
    assert packet["skipped_reads"][0]["tool_call_id"] == "missing"
