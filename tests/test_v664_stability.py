"""v6.6.4: batched C, the I/R/S tool-output cap and the submission filter (no Docker, no model)."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents import mini_v66_review
from external_agents_eval.agents.mini_v62 import MiniBashEnvironment, model_visible_events
from external_agents_eval.agents.mini_v66_review import (
    OUTPUT_CAP_CHARS,
    V66FileReviewEnvironment,
    V66FrozenReviewRunner,
    V66SummaryBashEnvironment,
    cap_output,
)
from external_agents_eval.benchmarks.deepswe.submission_filter import is_test_path, strip_added_tests
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v66_checkpoints import (
    CheckpointClient,
    check_batch,
    finalize_batch,
    target_batches,
)
from external_agents_eval.workflows.solve_review_repair.v66_prompts import (
    FACETS,
    checkpoint_system_prompt,
    checkpoint_user,
)

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_LINES = [f"line {n}\n" for n in range(40, 53)]
CONFIG_LINES[4] = "    key, value = (part.strip() for part in line.split('=', 1))\n"  # line 44
PACKET = {"task": {"description": "parse_env returns a dict of KEY=VALUE lines.", "title": "Env"},
          "managed_delivery": [],
          "evidence": [{"path": "config.py", "line_start": 40, "line_end": 52, "text": "".join(CONFIG_LINES)}]}


def ledger(count, preserved=0):
    def item(prefix, n):
        return {"id": f"{prefix}{n:03d}", "content": {"statement": f"statement {prefix}{n}",
                                                       "importance": "required", "sources": []}}
    return {"summary": "s", "obligations": [item("O", n) for n in range(1, count + 1)],
            "preservation_obligations": [item("P", n) for n in range(1, preserved + 1)],
            "open_questions": []}


def point(target, facet="D2", resolution="analog", **overrides):
    value = {"targets": [target], "facet": facet, "given": f"given {target} {facet}", "when": "w",
             "then": f"then {target} {facet}", "resolution": resolution,
             "strength": {"task": "required", "open": "open"}.get(resolution, "inferred"),
             "rules_out": None if resolution == "open" else "keeping spaces",
             "sources": [{"kind": "repository", "path": "config.py", "line_start": 44, "line_end": 44}],
             "basis_quote": "part.strip() for part in line.split('=', 1)",
             "basis_behavior": "strips spaces", "convention": None, "alternatives": [], "favored": None}
    if resolution == "open":
        value.update(sources=[], basis_quote=None, basis_behavior=None, alternatives=["skip", "raise"])
    value.update(overrides)
    return value


def payload(points, targets):
    covered = {(t, p["facet"]) for p in points for t in p["targets"]}
    return {"checkpoints": points, "coverage": [
        {"target": t, "facet": f, "status": "covered" if (t, f) in covered else "not_applicable",
         "reason": None if (t, f) in covered else "no divergence"}
        for t in targets for f in FACETS]}


# ---------------------------------------------------------------- C batches

def test_targets_are_split_in_ledger_order_into_even_batches_of_at_most_eight():
    sizes = lambda n, p=0: [len(b) for b in target_batches(ledger(n, p), 8)]  # noqa: E731
    assert sizes(25) == [7, 6, 6, 6] and sizes(27, 11) == [8, 8, 8, 7, 7]
    assert sizes(8) == [8] and sizes(9) == [5, 4] and target_batches(ledger(0), 8) == []
    assert target_batches(ledger(2, 1), 8) == [["O001", "O002", "P001"]]


def test_batch_prompts_share_the_long_prefix_and_end_with_the_batch_targets():
    first = checkpoint_user(copy.deepcopy(PACKET), ledger(3), (["O001", "O002"], 1, 2))
    second = checkpoint_user(copy.deepcopy(PACKET), ledger(3), (["O003"], 2, 2))
    prefix = checkpoint_user(copy.deepcopy(PACKET), ledger(3))
    assert first.startswith(prefix.rstrip("\n")) and second.startswith(prefix.rstrip("\n"))
    assert first.endswith("A check point lists only targets of this batch.")
    assert "CURRENT BATCH 2 of 2\nTargets in this batch: O003" in second
    system = checkpoint_system_prompt(4)
    assert "For every O and P target in the current batch" in system
    assert "coverage must list every target in the current batch" in system


def test_violations_are_classified_by_point_batch_and_structure():
    scope = ["O001", "O002"]
    data = payload([point("O001"), point("O002", alternatives=["a", "b"]), point("O003")], scope)
    data["coverage"].pop()
    data["notes"] = "extra"
    report = check_batch(data, ledger=ledger(3), packet=PACKET, max_per_target=4, batch=scope)
    assert not report.structure and sorted(report.points) == [1, 2]
    assert "only open check points" in report.points[1][0]
    assert "outside this batch: O003" in report.points[2][0]
    assert any("missing O002/D8" in e for e in report.batch)
    assert any("notes" in e for e in report.batch)  # an extra key is not fatal
    bad_item = payload([point("O001", facet="D9")], scope)
    assert list(check_batch(bad_item, ledger=ledger(2), packet=PACKET, max_per_target=4).points) == [0]
    for broken in ([], {"coverage": []}, {"checkpoints": {}}):
        assert check_batch(broken, ledger=ledger(2), packet=PACKET, max_per_target=4).structure


def test_finalize_drops_invalid_points_merges_trims_and_rebuilds_coverage():
    scope = ["O001", "O002"]
    points = [point("O001"), point("O002", alternatives=["a", "b"]),
              point("O002", facet="D4", given="given O001 D4", then="then O001 D4"),
              point("O001", facet="D4"),
              *[point("O001", facet=f) for f in ("D5", "D6", "D7")]]
    data = payload(points, scope)
    report = check_batch(data, ledger=ledger(2), packet=PACKET, max_per_target=4, batch=scope)
    kept, coverage, outcome = finalize_batch(data, report, scope=scope, max_per_target=4)
    assert outcome["status"] == "degraded" and [d["index"] for d in outcome["dropped"]] == [1]
    assert outcome["merged"] and kept[1]["targets"] == ["O002", "O001"]
    assert [p["facet"] for p in kept] == ["D2", "D4", "D5", "D6"]  # O001 reached 4 before D7
    assert outcome["trimmed"] == [{"facet": "D7", "targets_removed": ["O001"], "point_dropped": True}]
    assert len(coverage) == 16
    cell = {(c["target"], c["facet"]): c for c in coverage}
    assert cell[("O002", "D2")]["status"] == "uncovered"  # its only point was dropped
    assert cell[("O001", "D7")]["status"] == "uncovered"
    assert cell[("O002", "D1")] == {"target": "O002", "facet": "D1", "status": "not_applicable",
                                    "reason": "no divergence"}
    failed = finalize_batch(None, check_batch([], ledger=ledger(2), packet=PACKET, max_per_target=4),
                            scope=scope, max_per_target=4)
    assert failed[0] == [] and failed[2]["status"] == "failed"
    assert {c["status"] for c in failed[1]} == {"uncovered"}


def response(content, finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": "r"},
                         "finish_reason": finish}], "usage": {"completion_tokens": 1}}


def test_client_runs_each_batch_with_one_correction_and_keeps_the_valid_points(tmp_path, monkeypatch):
    the_ledger = ledger(9)  # batches of 4 -> O001-O003 | O004-O006 | O007-O009
    client = CheckpointClient(model="m", base_url="http://x", api_key="k", max_per_target=4,
                              max_attempts=2, timeout_sec=5, batch_max_targets=4)
    scopes = target_batches(the_ledger, 4)
    assert [len(s) for s in scopes] == [3, 3, 3]
    good1 = payload([point("O001"), point("O002", facet="D3", given="an empty header block")], scopes[0])
    bad2 = payload([point("O004", facet="D3", given="a header with trailing tabs and spaces"),
                    point("O005", alternatives=["a", "b"])], scopes[1])
    # Same rule as batch 1's first point, for O007: merged across batches.
    twin = point("O007", given="given O001 D2", then="then O001 D2")
    replies = {1: [response("{"), response(json.dumps(good1))],
               2: [response(json.dumps(bad2)), response(json.dumps(bad2))],
               3: [response("x" * 10, "length"), response(json.dumps(payload([twin], scopes[2])))]}
    sent = []

    def post(messages):
        sent.append(messages)
        user = messages[1]["content"]
        index = int(user.split("CURRENT BATCH ")[1].split(" ")[0])
        return replies[index].pop(0)

    monkeypatch.setattr(client, "_post", post)
    result = asyncio.run(client.generate(packet=copy.deepcopy(PACKET), ledger=the_ledger,
                                         artifacts=ArtifactStore(tmp_path)))
    assert result["status"] == "passed" and len(result["attempts"]) == 6
    assert [b["status"] for b in result["batches"]] == ["valid", "degraded", "valid"]
    assert result["degraded_batches"] == [2] and result["failed_batches"] == []
    assert len(sent) == 6 and all(m[0]["content"] == sent[0][0]["content"] for m in sent)
    assert "violates the contract" in sent[1][-1]["content"] and len(sent[1]) == 4
    saved = json.loads((tmp_path / "v6.6/checkpoints/checkpoints.json").read_text())
    assert [p["targets"] for p in saved["checkpoints"]] == [["O001", "O007"], ["O002"], ["O004"]]
    assert result["cross_batch_merged"][0]["from_targets"] == ["O007"]
    assert len(saved["coverage"]) == 9 * 8
    assert result["summary"]["dropped_points"] == 1 and result["summary"]["batches"] == 3
    assert result["summary"]["uncovered_cells"] == 1  # O005/D2
    assert (tmp_path / "v6.6/checkpoints/batch-02/final.json").exists()
    with pytest.raises(RuntimeError, match="resampling"):
        asyncio.run(client.generate(packet=copy.deepcopy(PACKET), ledger=the_ledger,
                                    artifacts=ArtifactStore(tmp_path)))


def test_a_batch_that_never_parses_fails_alone_and_no_points_means_invalid(tmp_path, monkeypatch):
    client = CheckpointClient(model="m", base_url="http://x", api_key="k", max_per_target=4,
                              max_attempts=2, timeout_sec=5, batch_max_targets=2)
    the_ledger = ledger(4)
    replies = [response(json.dumps(payload([point("O001")], ["O001", "O002"]))), response("{"), response("{")]
    monkeypatch.setattr(client, "_post", lambda messages: replies.pop(0))
    result = asyncio.run(client.generate(packet=copy.deepcopy(PACKET), ledger=the_ledger,
                                         artifacts=ArtifactStore(tmp_path / "a")))
    assert result["status"] == "passed" and result["failed_batches"] == [2]
    assert result["summary"]["uncovered_cells"] == 16  # batch 1 marked the rest not_applicable
    monkeypatch.setattr(client, "_post", lambda messages: response("{}"))
    client.transport = []
    result = asyncio.run(client.generate(packet=copy.deepcopy(PACKET), ledger=the_ledger,
                                         artifacts=ArtifactStore(tmp_path / "b")))
    assert result["status"] == "invalid_output"
    assert not (tmp_path / "b/v6.6/checkpoints/checkpoints.json").exists()


# ---------------------------------------------------------------- output cap

def test_cap_keeps_head_and_tail_and_never_touches_a_submission():
    text = "H" * 20_000 + "M" * 50_000 + "T" * 20_000
    capped = cap_output(text)
    assert capped.startswith("H" * 15_000) and capped.endswith("T" * 15_000) and "M" not in capped
    assert "60000 of 90000 characters omitted" in capped
    assert cap_output("x" * OUTPUT_CAP_CHARS) == "x" * OUTPUT_CAP_CHARS
    submission = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + "{" * 40_000
    assert cap_output(submission) == submission


class Bridge:
    def __init__(self, stdout):
        self.stdout = stdout

    def exec_sync(self, command, cwd=None):
        return SimpleNamespace(stdout=self.stdout, stderr="", return_code=0)


def test_summary_environment_records_the_capped_output_as_the_evidence():
    environment = V66SummaryBashEnvironment(Bridge("z" * 100_000), "/app")
    result = environment.execute({"command": "grep -R x .", "tool_call_id": "c1"})
    assert len(result["output"]) < 31_000
    messages = [{"role": "tool", "tool_call_id": "c1", "content": result["output"]}]
    visible, unreturned = model_visible_events(environment.events, messages)
    assert len(visible) == 1 and not unreturned
    plain = MiniBashEnvironment(Bridge("z" * 100_000), "/app")
    assert len(plain.execute({"command": "grep -R x ."})["output"]) == 100_000  # V6.2/V6.3 unchanged


def test_review_environments_are_capped_for_i_r_and_s_only(tmp_path, monkeypatch):
    raw = SimpleNamespace(stdout="a" * 50_000, stderr="b" * 10)
    environment = object.__new__(V66FileReviewEnvironment)
    assert len(environment._command_output(raw)) < 31_000
    runner = object.__new__(V66FrozenReviewRunner)
    runner.environment = Bridge("")
    parent = V66FrozenReviewRunner.__mro__[1]
    monkeypatch.setattr(parent, "_make_mini_environment",
                        lambda self, **kw: MiniBashEnvironment(self.environment, kw["worktree"], tools=kw["tools"]))
    common = {"prompt": "p", "worktree": "/app", "tools": None, "schemas": None, "host": tmp_path}
    summary = runner._make_mini_environment(role=RoleName.V5_SUMMARY_REVIEWER, **common)
    repair = runner._make_mini_environment(role=RoleName.REPAIR, **common)
    assert type(summary) is V66SummaryBashEnvironment and type(repair) is MiniBashEnvironment
    assert mini_v66_review.OUTPUT_HEAD_CHARS + mini_v66_review.OUTPUT_TAIL_CHARS == OUTPUT_CAP_CHARS


# ---------------------------------------------------------------- submission filter

def section(path, body="+x\n", new=True):
    header = f"diff --git a/{path} b/{path}\n"
    if new:
        return header + f"new file mode 100644\nindex 0000000..1111111\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n{body}"
    return header + f"index 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-y\n{body}"


def test_only_newly_added_test_files_are_removed():
    patch = "".join([
        section("pkg/parse.go"),
        section("pkg/parse_test.go"),
        section("pkg/existing_test.go", new=False),
        section("tests/data/101_ddl.sql"),
        section("src/latest.ts"),
        section("packages/core/test/prompts/path.test.ts"),
        section("src/util.spec.ts"),
        section("conftest.py"),
        section("internal/testdata/golden.txt"),
        "diff --git a/tests/fixture.bin b/tests/fixture.bin\nnew file mode 100644\nindex 0000000..3333333\n"
        "GIT binary patch\nliteral 3\nKcmZ>Y%K!iX\n\nliteral 0\nHcmV?d00001\n\n",
    ])
    submitted, removed = strip_added_tests(patch)
    assert removed == ["pkg/parse_test.go", "tests/data/101_ddl.sql", "packages/core/test/prompts/path.test.ts",
                       "src/util.spec.ts", "conftest.py", "internal/testdata/golden.txt", "tests/fixture.bin"]
    assert submitted == section("pkg/parse.go") + section("pkg/existing_test.go", new=False) + section("src/latest.ts")
    assert strip_added_tests("") == ("", [])
    assert not is_test_path("src/contest.py") and not is_test_path("lib/testing.go")
    assert is_test_path("src/test/java/FooTest.java") and is_test_path("spec/models/user_spec.rb")


def test_e2e_runner_grades_the_filtered_patch_and_freezes_the_new_sources():
    spec = importlib.util.spec_from_file_location("v664e2e", PROJECT / "scripts/deepswe_v66_e2e/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    for name in ("external_agents_eval/benchmarks/deepswe/submission_filter.py",
                 "external_agents_eval/agents/mini_v62.py", "external_agents_eval/agents/mini_v63.py"):
        assert name in run.FROZEN_SOURCES
    source = (PROJECT / "scripts/deepswe_v66_e2e/run.py").read_text()
    assert source.count("submitted, removed = submission(") == 2
    assert "native.grade(root, augmented, image.id, submitted," in source
    assert "native.grade(root, task, image.id, submitted," in source


def test_regrade_rows_mark_only_patches_the_filter_changes(tmp_path):
    spec = importlib.util.spec_from_file_location("regrade", PROJECT / "scripts/deepswe_submission_filter_regrade/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    for tid, patch, reward in (("a", section("x_test.go"), 0), ("b", section("x.go"), 1)):
        (tmp_path / "outputs" / tid).mkdir(parents=True)
        (tmp_path / "outputs" / tid / "model.patch").write_text(patch)
        (tmp_path / "outputs" / tid / "result.json").write_text(json.dumps({"status": "scored", "reward": reward}))
    (tmp_path / "outputs/c").mkdir()  # no patch: skipped
    rows = run.rows_for("m0", tmp_path)
    assert [(r["task_id"], r["changed"], r["removed"]) for r in rows] == [("a", True, ["x_test.go"]),
                                                                          ("b", False, [])]


def test_c_runner_runs_tasks_in_parallel_with_the_batch_size(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("v664c", PROJECT / "scripts/deepswe_v66_c/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    source = tmp_path / "source"
    for tid in ("t1", "t2"):
        (source / "a" / tid / "v6_review_repair/v6.2").mkdir(parents=True)
        (source / "a" / tid / "ledger.json").write_text(json.dumps(ledger(3)))
        (source / "a" / tid / "v6_review_repair/v6.2/evidence.json").write_text(
            json.dumps({**PACKET, "task": {**PACKET["task"], "extra": 1}}))
    seen = []

    class Client:
        def __init__(self, **kwargs):
            seen.append(kwargs["batch_max_targets"])

        async def generate(self, *, packet, ledger, artifacts):
            assert set(packet["task"]) == {"description", "title"}
            return {"status": "passed", "attempts": [], "summary": {"checkpoints": 1},
                    "degraded_batches": [], "failed_batches": []}

    monkeypatch.setattr(run, "CheckpointClient", Client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    selection = run.prepare(tmp_path / "root", source, ["t1", "t2"], "test")
    assert selection["config"]["batch_max_targets"] == 8
    run.run(tmp_path / "root", selection, workers=2)
    assert seen == [8, 8]
    for tid in ("t1", "t2"):
        assert json.loads((tmp_path / "root/a" / tid / "a-result.json").read_text())["status"] == "checkpoints_ready"
