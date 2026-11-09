"""Exercise public absolute paths through real Pi reads and the frozen FT collector."""
from __future__ import annotations

import asyncio
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from external_agents_eval.agents import mini_v62
from external_agents_eval.agents.pi_file_tools import PI_EVIDENCE_PROTOCOL, runtime_directory
from external_agents_eval.benchmarks.featurebench.agents import FeatureBenchPiFileEnvironment
from external_agents_eval.workflows.solve_review_repair.ft_evidence import observed_evidence


def test_absolute_and_relative_reads_retain_all_frozen_source_and_provenance(tmp_path, monkeypatch):
    runtime = runtime_directory()
    if not (runtime / "bin/node").exists():
        pytest.skip("Pinned Linux Pi runtime required for the actual tool integration")
    root = tmp_path / "a-worktree"
    (root / "src").mkdir(parents=True)
    source = "\n".join(f"源代码 {line}" for line in range(1, 11)) + "\n"
    (root / "src/api.py").write_text(source)

    def execute(bridge, cwd, name, arguments, call_id):
        request = {"cwd": cwd, "name": name, "arguments": arguments, "tool_call_id": call_id}
        return json.loads(subprocess.check_output([
            str(runtime / "bin/node"), str(Path(mini_v62.__file__).with_name("pi_file_tools.mjs")),
            str(runtime), json.dumps(request),
        ], text=True))

    monkeypatch.setattr(mini_v62, "execute_tool", execute)
    environment = FeatureBenchPiFileEnvironment(None, str(root))
    actions = [
        {"name": "read", "arguments": {"path": "src/api.py", "offset": 1, "limit": 5},
         "tool_call_id": "relative"},
        {"name": "read", "arguments": {"path": "/testbed/src/api.py", "offset": 4, "limit": 7},
         "tool_call_id": "absolute"},
    ]
    originals = deepcopy(actions)
    for action in actions:
        assert environment.execute(action)["returncode"] == 0
    assert actions == originals  # The raw model request remains auditable.
    assert environment.events[1][1]["featurebench_requested_path"] == "/testbed/src/api.py"
    assert environment.events[1][1]["path"] == str(root / "src/api.py")

    steps = []
    for action, (name, arguments, result) in zip(actions, environment.events, strict=True):
        steps.append({
            "step_id": len(steps) + 1,
            "tool_calls": [{"tool_call_id": action["tool_call_id"], "function_name": name,
                            "arguments": arguments}],
            "observation": {"results": [{"source_call_id": action["tool_call_id"],
                                          "content": result["output"], "exit_code": 0}]},
        })

    class FrozenWorkspace:
        async def read_frozen_file(self, *, path, **kwargs):
            assert path == "src/api.py"
            return path, source.encode()

    packet = asyncio.run(observed_evidence(
        {"evidence_protocol": PI_EVIDENCE_PROTOCOL, "steps": steps}, task_id="featurebench-probe",
        base_root=str(root), base_commit="frozen", workspace=FrozenWorkspace(),
    ))
    assert packet["skipped_reads"] == []
    assert len(packet["read_provenance"]) == 2
    assert len(packet["evidence"]) == 1
    assert packet["evidence"][0]["text"] == source
    assert packet["evidence"][0]["line_start"] == 1
    assert packet["evidence"][0]["line_end"] == 10
