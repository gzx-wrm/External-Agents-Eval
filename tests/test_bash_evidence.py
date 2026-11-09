from __future__ import annotations

import asyncio
import copy
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.workflows.solve_review_repair.bash_evidence import PROTOCOL
from external_agents_eval.workflows.solve_review_repair.ft_evidence import observed_evidence
from external_agents_eval.workflows.solve_review_repair.ft_protocol import render_user

SOURCE = "# baseline source\n\ndef feature(value):\n    result = value + 1\n    return result\n\n# end of source\n"


class Workspace:
    def __init__(self, files):
        self.files = files

    async def read_frozen_text_files(self, **kwargs):
        return self.files

    async def read_frozen_file(self, *, path, **kwargs):
        return (path, self.files[path]) if path in self.files else None


def trajectory(command, output, *, native=True, arguments=None):
    return {"evidence_protocol": PROTOCOL if native else "pi", "steps": [{
        "step_id": 2,
        "tool_calls": [{"tool_call_id": "call-1", "function_name": "bash" if native else "read",
                        "arguments": {"command": command, "cwd": "/base"} if native else arguments}],
        "observation": {"results": [{"source_call_id": "call-1", "content": output}]},
    }]}


def packet(data, files):
    return asyncio.run(observed_evidence(data, task_id="task", base_root="/base",
                                         base_commit="baseline", workspace=Workspace(files)))


@pytest.mark.parametrize("command,first,last", [
    ("cat source.py", 1, 7),
    ("head -n 4 source.py", 1, 4),
    ("tail -n 3 source.py", 5, 7),
    ("sed -n '3,5p' source.py", 3, 5),
    ("cd /base && sed -n '3,5p' source.py", 3, 5),
    ("bash -lc \"sed -n '3,5p' source.py\"", 3, 5),
    ("cat source.py | head -n 4", 1, 4),
    ("nl -ba source.py | sed -n '3,5p'", 3, 5),
    ("cat -n source.py | sed -n '3,5p'", 3, 5),
    ("SOURCE=source.py; sed -n '3,5p' \"$SOURCE\"", 3, 5),
    ("python -c \"from pathlib import Path; print(Path('source.py').read_text(), end='')\"", 1, 7),
])
def test_native_shell_returns_exactly_the_same_ft_evidence_as_pi(tmp_path, command, first, last):
    (tmp_path / "source.py").write_text(SOURCE)
    actual = command.replace("cd /base", "cd " + shlex.quote(str(tmp_path)))
    result = subprocess.run(actual, shell=True, executable="/bin/bash", cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    files = {"source.py": SOURCE.encode()}
    native = packet(trajectory(command, result.stdout + result.stderr), files)
    body = "\n".join(SOURCE.splitlines()[first - 1:last])
    pi = packet(trajectory("", body, native=False,
                          arguments={"path": "/base/source.py", "offset": first,
                                     "limit": last - first + 1}), files)
    assert native["evidence"] == pi["evidence"]
    assert not native["bash_evidence_audit"]["errors"]
    record = {"task": {"title": "Task", "description": "Issue"}, "managed_delivery": []}
    assert render_user(record, native["evidence"]) == render_user(record, pi["evidence"])


def test_combined_commands_collect_all_files_and_merge_only_visible_adjacent_ranges():
    files = {"a.py": b"a1\na2\na3\na4\n", "b.py": b"b1\nb2\nb3\n"}
    data = trajectory("sed -n '1,2p' a.py; echo '---'; sed -n '4p' a.py; cat b.py",
                      "a1\na2\n---\na4\nb1\nb2\nb3\n")
    result = packet(data, files)
    assert [(e["path"], e["line_start"], e["line_end"]) for e in result["evidence"]] == [
        ("a.py", 1, 2), ("a.py", 4, 4), ("b.py", 1, 3)]
    assert not result["bash_evidence_audit"]["errors"]


@pytest.mark.parametrize("command,output", [
    ("ls", "source.py\n"),
    ("find . -name '*.py'", "./source.py\n"),
    ("grep -n feature source.py", "3:def feature(value):\n"),
    ("rg feature source.py | head -10", "def feature(value):\n"),
])
def test_directory_and_search_output_stays_discovery_like_historical_ft(command, output):
    result = packet(trajectory(command, output), {"source.py": SOURCE.encode()})
    assert result["evidence"] == []
    assert len(result["repository_inventory_provenance"]) == 1


def test_output_truncation_cannot_add_unseen_source_lines():
    result = packet(trajectory("cat source.py", "# baseline source\n\ndef feature(val"),
                    {"source.py": SOURCE.encode()})
    assert result["evidence"][0]["text"] == "# baseline source\n\n"
    assert result["evidence"][0]["line_end"] == 2


def test_unknown_python_reader_can_be_mapped_from_unique_output_without_path_hints():
    source = b"def distinctive_feature():\n    return 18273\n"
    result = packet(trajectory("python inspect_repository.py", source.decode()), {"pkg/api.py": source})
    assert result["evidence"][0]["path"] == "pkg/api.py"
    assert not result["bash_evidence_audit"]["errors"]


def test_single_line_from_an_indirect_reader_is_not_silently_discarded():
    source = b"unique_source_line_18273\n"
    result = packet(trajectory("python inspect_repository.py", source.decode()), {"pkg/api.py": source})
    assert result["evidence"][0]["text"] == source.decode()


def test_ambiguous_single_line_is_explicitly_reported():
    result = packet(trajectory("python inspect_repository.py", "shared_line\n"),
                    {"a.py": b"shared_line\n", "b.py": b"shared_line\n"})
    assert result["bash_evidence_audit"]["errors"]


def test_ambiguous_output_is_reported_instead_of_silently_choosing_or_omitting_a_file():
    source = b"def duplicate():\n    return 1\n"
    result = packet(trajectory("python inspect_repository.py", source.decode()),
                    {"a.py": source, "b.py": source})
    assert result["bash_evidence_audit"]["errors"][0]["status"] == "ambiguous"


def test_modified_read_is_reported_and_cannot_be_accepted_as_complete():
    result = packet(trajectory("cat source.py", SOURCE.replace("value + 1", "value + 2")),
                    {"source.py": SOURCE.encode()})
    assert result["bash_evidence_audit"]["errors"][0]["status"] == "unmapped_read"


@pytest.mark.parametrize("command,separator", [("cat a.py b.py", ""),
                                              ("cat a.py; cat b.py", ""),
                                              ("cat a.py; echo '---'; cat b.py", "---\n")])
def test_identical_files_in_one_cat_are_both_preserved(command, separator):
    source = b"def same():\n    return 1\n"
    result = packet(trajectory(command, source.decode() + separator + source.decode()),
                    {"a.py": source, "b.py": source})
    assert [e["path"] for e in result["evidence"]] == ["a.py", "b.py"]
    assert not result["bash_evidence_audit"]["errors"]


def test_numbered_cat_across_files_keeps_both_source_locations():
    result = packet(trajectory("cat -n a.py b.py", "     1\talpha\n     2\tbeta\n"),
                    {"a.py": b"alpha\n", "b.py": b"beta\n"})
    assert [e["path"] for e in result["evidence"]] == ["a.py", "b.py"]


def test_final_draft_is_never_mistaken_for_repository_evidence():
    output = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + SOURCE
    result = packet(trajectory("submit_json 'draft'", output), {"source.py": SOURCE.encode()})
    assert not result["evidence"]
    assert result["bash_evidence_audit"]["calls"][0]["status"] == "delivery"


def test_transformed_read_without_matching_source_is_explicitly_reported():
    result = packet(trajectory("python -c \"print(open('source.py').read().upper())\"", SOURCE.upper()),
                    {"source.py": SOURCE.encode()})
    assert result["bash_evidence_audit"]["errors"]


def test_directory_listing_with_stderr_redirect_is_still_only_discovery():
    result = packet(trajectory("ls 2>/dev/null | head -10", "source.py\n"), {"source.py": SOURCE.encode()})
    assert result["bash_evidence_audit"]["calls"][0]["status"] == "discovery"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_incomplete_mapping_is_archived_and_stops_before_ft(tmp_path, ambiguous):
    from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore
    from external_agents_eval.workflows.solve_review_repair.models import RoleRunResult, TaskInput
    from external_agents_eval.workflows.solve_review_repair.v62_controller import V62Controller

    raw = b"def same():\n    return True\n"
    files = {"a.py": raw, "b.py": raw} if ambiguous else {"a.py": raw}
    command = "python inspector.py" if ambiguous else "python -c \"print(open('a.py').read().upper())\""
    output = raw.decode() if ambiguous else raw.decode().upper()

    class FrozenWorkspace(Workspace):
        async def tracked_diff_digest(self, **kwargs):
            return "unchanged"

    class Search:
        async def run_role(self, **kwargs):
            path = tmp_path / "search.json"
            path.write_text(json.dumps(trajectory(command, output)))
            return RoleRunResult(exit_code=0, final_assistant_text="discarded draft",
                                 transcript_path=path, trajectory_path=path,
                                 started_at="now", finished_at="now")

    async def forbidden_ft(*args, **kwargs):
        raise AssertionError("FT must not run with missing evidence")

    artifacts = ArtifactStore(tmp_path / "artifacts")
    controller = V62Controller(role_runner=Search(), ledger_client=SimpleNamespace(generate=forbidden_ft),
                               workspace=FrozenWorkspace(files), artifacts=artifacts)
    try:
        task = TaskInput(task_id="task", repository_name="repo", base_commit="baseline", issue_body="Fix it.")
        assert asyncio.run(controller.prepare_ledger(task, base_worktree="/base")) is None
        assert "mapping is incomplete" in controller.ledger_error
        assert json.loads((artifacts.root / "v6.2/bash-evidence-audit.json").read_text())["errors"]
        assert not (artifacts.root / "v6.2/ledger/request.json").exists()
    finally:
        controller.close()


ROOT = Path(__file__).resolve().parents[1]
RECORDS = sorted((ROOT / "a_agent_finetune/data/f_teacher_observed_30_current/records").glob("*.json"))


@pytest.mark.parametrize("record_path", RECORDS, ids=lambda p: p.stem)
def test_historical_30_pi_reads_and_equivalent_bash_returns_have_identical_ft_inputs(record_path):
    """Replay historical observations, including byte caps, EOF, aliases and the Numba exclusion."""
    record = json.loads(record_path.read_text())
    repo = ROOT / "a_agent_finetune/.cache/baselines" / record["task_id"]
    if not (repo / ".git").exists():
        pytest.skip("Historical frozen repository is not cached")
    revision = record["base_commit"]
    aliases = {r["requested_path"]: r["path"] for r in record["read_provenance"]}
    files = {}
    for path in sorted({e["path"] for e in record["evidence"]}):
        files[path] = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])
    for requested, canonical in aliases.items():
        if canonical in files:
            files[requested] = files[canonical]

    class HistoricalWorkspace(Workspace):
        async def read_frozen_file(self, *, path, **kwargs):
            canonical = aliases.get(path, path)
            return (canonical, self.files[canonical]) if canonical in self.files else None

    original = json.loads((ROOT / record["provenance"]["a_trajectory"]).read_text())
    prompt = (ROOT / record["provenance"]["spec_prompt"]).read_text()
    spec, _ = json.JSONDecoder().raw_decode(prompt.split("SPEC INPUT\n", 1)[1].lstrip())
    base_root = spec["base_root"]
    converted = {"evidence_protocol": PROTOCOL, "steps": []}
    for step in original["steps"]:
        for call in step.get("tool_calls", []):
            if call["function_name"] != "read":
                continue
            args = call["arguments"]
            response = next(r for r in step["observation"]["results"]
                            if r["source_call_id"] == call["tool_call_id"])
            new = copy.deepcopy(call)
            if args.get("path", "").startswith(base_root.rstrip("/") + "/"):
                first = int(args.get("offset", 1))
                last = first + int(args.get("limit", 2**31)) - 1
                new.update(function_name="bash", arguments={
                    "command": f"sed -n '{first},{last}p' " + shlex.quote(args["path"]),
                    "cwd": base_root,
                })
            converted["steps"].append({"step_id": step["step_id"], "tool_calls": [new],
                                       "observation": {"results": [response]}})
    kwargs = {"task_id": record["task_id"], "base_root": base_root, "base_commit": revision,
              "workspace": HistoricalWorkspace(files)}
    pi = asyncio.run(observed_evidence(original, **kwargs))
    structured = asyncio.run(observed_evidence({**original, "evidence_protocol": "pi-read-observed-v1"}, **kwargs))
    bash = asyncio.run(observed_evidence(converted, **kwargs))
    assert structured["evidence"] == bash["evidence"] == pi["evidence"] == record["evidence"]
    assert "bash_evidence_audit" not in structured
    assert not bash["bash_evidence_audit"]["errors"]
    assert render_user(record, bash["evidence"]) == render_user(record, pi["evidence"])
    assert render_user(record, structured["evidence"]) == render_user(record, pi["evidence"])
