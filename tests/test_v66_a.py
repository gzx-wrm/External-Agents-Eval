"""V6.6 A pieces that need neither Docker nor model calls."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore
from external_agents_eval.workflows.solve_review_repair.models import RoleName, RoleRunResult
from external_agents_eval.workflows.solve_review_repair.v62_controller import V62Controller
from external_agents_eval.workflows.solve_review_repair.v66_checkpoints import (
    CheckpointClient,
    delivery_views,
    validate_checkpoints,
)
from external_agents_eval.workflows.solve_review_repair.v66_config import V66Config
from external_agents_eval.workflows.solve_review_repair.v66_controller import (
    PiApiFailure,
    V66Controller,
    evidence_plan_gaps,
    merge_trajectories,
    pi_completion,
)
from external_agents_eval.workflows.solve_review_repair.v66_prompts import (
    EVIDENCE_PLAN,
    FACETS,
    checkpoint_system_prompt,
    checkpoint_user,
    evidence_plan_continuation,
    evidence_plan_prompt,
)

PROJECT = Path(__file__).resolve().parents[1]
LEDGER = {"summary": "s",
          "obligations": [
              {"id": "O001", "content": {"statement": "parse_env returns a dict", "importance": "required",
                                         "sources": [{"kind": "task", "field": "description",
                                                      "quote": "returns a dict"}]}},
              {"id": "O002", "content": {"statement": "dump_env writes lines", "importance": "required",
                                         "sources": [{"kind": "task", "field": "description",
                                                      "quote": "KEY=VALUE lines"}]}}],
          "preservation_obligations": [], "open_questions": []}
CONFIG_LINES = [f"line {n}\n" for n in range(40, 53)]
CONFIG_LINES[4] = "    key, value = (part.strip() for part in line.split('=', 1))\n"  # line 44
PACKET = {"task": {"description": "parse_env returns a dict of KEY=VALUE lines.", "title": "Env"},
          "managed_delivery": [],
          "evidence": [{"path": "config.py", "line_start": 40, "line_end": 52, "text": "".join(CONFIG_LINES)}]}


def point(facet="D2", resolution="analog", targets=("O001",), **overrides):
    value = {"targets": list(targets), "facet": facet, "given": f"g {facet}", "when": "w",
             "then": f"t {facet} {resolution}", "resolution": resolution,
             "strength": {"task": "required", "open": "open"}.get(resolution, "inferred"),
             "rules_out": None if resolution == "open" else "keeping spaces",
             "sources": [{"kind": "repository", "path": "config.py", "line_start": 44, "line_end": 44}],
             "basis_quote": "part.strip() for part in line.split('=', 1)",
             "basis_behavior": "strips spaces", "convention": None, "alternatives": [], "favored": None}
    if resolution == "task":
        value.update(sources=[{"kind": "task", "field": "description", "quote": "KEY=VALUE lines"}],
                     basis_quote=None, basis_behavior=None)
    if resolution == "open":
        value.update(sources=[], basis_quote=None, basis_behavior=None, alternatives=["skip", "raise"])
    if resolution == "convention":
        value.update(sources=[], basis_quote=None, convention="Python dict semantics",
                     basis_behavior="later keys win")
    value.update(overrides)
    return value


def payload(points):
    covered = {(t, p["facet"]) for p in points for t in p["targets"]}
    return {"checkpoints": points, "coverage": [
        {"target": t, "facet": f, "status": "covered" if (t, f) in covered else "not_applicable",
         "reason": None if (t, f) in covered else "no divergence"}
        for t in ("O001", "O002") for f in FACETS]}


def errors(data, limit=4):
    return validate_checkpoints(data, ledger=LEDGER, packet=PACKET, max_per_target=limit)


def test_valid_checkpoints_pass():
    good = payload([point(), point("D1", "task", targets=("O001", "O002")),
                    point("D4", "open"), point("D3", "convention"),
                    point("D5", "open", targets=("O002",), favored="raise", sources=[
                        {"kind": "repository", "path": "config.py", "line_start": 44, "line_end": 44}],
                        basis_quote="line.split('=', 1)", basis_behavior="splits once")])
    assert errors(good) == []


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["coverage"].pop(), "missing O002/D8"),
    (lambda p: p["coverage"][0].update(status="covered"), "is covered but has 0"),
    (lambda p: p["coverage"][0].update(reason=None), "needs a not_applicable reason"),
    (lambda p: p["checkpoints"][0].update(strength="required"), "requires strength inferred"),
    (lambda p: p["checkpoints"][0].update(sources=[]), "needs repository lines"),
    (lambda p: p["checkpoints"][0]["sources"][0].update(line_end=60), "outside the supplied evidence"),
    (lambda p: p["checkpoints"][0].update(targets=["O009"]), "unknown target"),
    (lambda p: p["checkpoints"][0].update(alternatives=["a", "b"]), "only open check points"),
    (lambda p: p["checkpoints"][0].update(rules_out=None), "name in rules_out"),
    (lambda p: p["checkpoints"][0].update(basis_quote="part.lower()"), "basis_quote is not exact"),
    (lambda p: p["checkpoints"][0].update(basis_quote=None), "copy the decisive text"),
    (lambda p: p["checkpoints"][0].update(basis_behavior=None), "basis_behavior"),
    (lambda p: p["checkpoints"][0].update(then="loadError stays unset or empty"), "merges omitted/null/empty"),
    (lambda p: p["checkpoints"][0].update(given="a null/empty value"), "merges omitted/null/empty"),
])
def test_contract_violations_are_reported(mutate, message):
    data = payload([point()])
    mutate(data)
    assert any(message in error for error in errors(data)), errors(data)


def test_open_points_need_alternatives_and_a_quoted_basis_for_a_favored_reading():
    data = payload([point("D4", "open", alternatives=["skip"], favored="raise")])
    found = errors(data)
    assert any("at least two alternatives" in e for e in found)
    assert any("favored must be one of" in e for e in found)
    favored_without_basis = payload([point("D4", "open", favored="raise")])
    assert any("copy the decisive text" in e for e in errors(favored_without_basis))
    convention_basis = payload([point("D4", "open", favored="raise", convention="HTTP folding",
                                      basis_behavior="replaces folds with SP")])
    assert errors(convention_basis) == []
    assert any("lists alternatives instead of rules_out" in e
               for e in errors(payload([point("D4", "open", rules_out="x")])))


def test_the_same_rule_for_two_targets_must_be_merged():
    twin = payload([point("D1", "task"), point("D1", "task", targets=("O002",))])
    assert any("merge them into one check point listing targets ['O001', 'O002']" in e for e in errors(twin))
    different_condition = payload([point("D1", "task", given="an empty file"),
                                   point("D1", "task", targets=("O002",), given="a file without newline")])
    assert errors(different_condition) == []


def test_task_quotes_must_be_exact_and_merged_points_count_for_each_target():
    data = payload([point("D1", "task", sources=[{"kind": "task", "field": "description",
                                                  "quote": "returns a mapping"}])])
    assert any("not exact" in e for e in errors(data))
    data = payload([point(f, "task", targets=("O001", "O002"), given=f"distinct {f}") for f in ("D1", "D2", "D3")])
    assert errors(data, limit=3) == []
    assert any("O002: 3 check points exceed" in e for e in errors(data, limit=2))


def test_delivery_views_rank_decision_points_and_cap_the_solver_view():
    points = [point("D1", "task"), point("D8", "task", targets=("O001", "O002")),
              point("D3", "convention"), point("D2"), point("D4", "open")]
    views = delivery_views(payload(points), max_solver_points=3)
    assert [p["resolution"] for p in views["solver_points"]] == ["open", "analog", "convention"]
    assert views["dropped_from_solver_view"] == 2 and len(views["review_targets"]) == 5
    assert "Candidate readings: skip | raise" in views["solver_text"]
    assert "Excludes: keeping spaces (reference, analog: strips spaces)" in views["solver_text"]


def test_prompts_contain_every_facet_and_no_benchmark_task():
    prompt = checkpoint_system_prompt(4) + EVIDENCE_PLAN + evidence_plan_continuation(["search", "test_read"])
    for key, (name, _) in FACETS.items():
        assert f"{key} {name}" in prompt
    assert "implicit input" in prompt and "three separate cases" in prompt and "rules_out" in prompt
    manifests = [PROJECT / "manifests/deepswe_v1.1_heldout30_from82_seed20261102.json",
                 PROJECT / "a_agent_finetune/eval30_current.json"]
    if not all(path.exists() for path in manifests):
        pytest.skip("task manifests are not part of this code snapshot")
    heldout, eval30 = (json.loads(path.read_text()) for path in manifests)
    names = set(heldout["task_ids"]) | set(eval30.get("task_ids", []))
    lowered = prompt.lower()
    assert not [name for name in names if name in lowered]
    assert not [name.split("-")[0] for name in names if name.split("-")[0] in lowered.split()]


def test_checkpoint_user_inserts_the_ledger_between_task_and_evidence():
    text = checkpoint_user(copy.deepcopy(PACKET), LEDGER)
    assert text.index("ORIGINAL TASK") < text.index("FROZEN REQUIREMENT LEDGER") \
        < text.index("FROZEN PUBLIC REPOSITORY EVIDENCE")
    assert '"id": "O001"' in text and "40 line 40" in text


def trajectory(*calls, prefix="c"):
    return {"steps": [{"step_id": n, "tool_calls": [{"tool_call_id": f"{prefix}{n}", "function_name": name,
                                                       "arguments": {"path": path}}]}
                      for n, (name, path) in enumerate(calls, 1)]}


def test_evidence_plan_gaps_use_search_calls_and_test_reads_inside_base_root():
    base = "/w/spec"
    assert evidence_plan_gaps(trajectory(("read", "/w/spec/src/a.py")), base_root=base) == ["search", "test_read"]
    assert evidence_plan_gaps(trajectory(("grep", "/w/spec"), ("read", "/w/spec/tests/test_a.py")),
                              base_root=base) == []
    for test_path in ("pkg/a_test.go", "src/__tests__/a.ts", "a.spec.ts", "test_a.py", "FooTest.java"):
        assert evidence_plan_gaps(trajectory(("find", "/w/spec"), ("read", f"/w/spec/{test_path}")),
                                  base_root=base) == [], test_path
    assert evidence_plan_gaps(trajectory(("find", "/w"), ("read", "/other/tests/test_a.py")),
                              base_root=base) == ["test_read"]


def test_merge_keeps_new_steps_once_and_renumbers():
    first = trajectory(("read", "/w/a.py"))
    second = trajectory(("read", "/w/a.py"), ("read", "/w/tests/test_a.py"))
    second["steps"][1]["tool_calls"][0]["tool_call_id"] = "new"
    merged = merge_trajectories(first, second)
    assert [s["step_id"] for s in merged["steps"]] == [1, 2]
    assert [s["tool_calls"][0]["arguments"]["path"] for s in merged["steps"]] == ["/w/a.py", "/w/tests/test_a.py"]


def events(*stops):
    return "\n".join(json.dumps({"type": "turn_end", "message": {"role": "assistant", "stopReason": s}})
                     for s in stops)


def test_pi_completion_requires_a_normal_final_turn():
    assert pi_completion(events("toolUse", "stop")) == {"last_stop_reason": "stop", "normal_completion": True}
    assert pi_completion(events("error", "error"))["normal_completion"] is False
    assert pi_completion(events("stop", "error"))["last_stop_reason"] == "error"
    assert pi_completion("") == {"last_stop_reason": None, "normal_completion": False}


def role_result(tmp_path, name, stops, calls):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "pi.jsonl").write_text(events(*stops))
    # Pi tool-call IDs are unique per session; the folder name keeps them distinct here.
    (folder / "trajectory.json").write_text(json.dumps(trajectory(*calls, prefix=name)))
    return RoleRunResult(exit_code=0, final_assistant_text="", transcript_path=folder / "pi.jsonl",
                         trajectory_path=folder / "trajectory.json", started_at="s", finished_at="f")


def controller(tmp_path):
    instance = object.__new__(V66Controller)
    instance.artifacts = ArtifactStore(tmp_path / "store")
    instance.config = V66Config()
    return instance


def run_search(instance, **kwargs):
    return asyncio.run(instance._run_role(
        role=RoleName.V5_SPECIFICATION_REVIEWER, prompt="BASE\n", worktree="/w/spec",
        prefix="v6.2/evidence-search", session_role="A", invocation_id="r.A.1", timeout_sec=1800, **kwargs))


def test_evidence_search_gets_the_plan_and_one_continuation_when_it_skipped_tests(tmp_path, monkeypatch):
    seen = []
    results = [role_result(tmp_path, "search", ["toolUse", "stop"], [("grep", "/w/spec"), ("read", "/w/spec/a.py")]),
               role_result(tmp_path, "cont", ["stop"], [("read", "/w/spec/tests/test_a.py")])]

    async def parent(self, **kwargs):
        seen.append(kwargs)
        return results.pop(0)

    monkeypatch.setattr(V62Controller, "_run_role", parent)
    instance = controller(tmp_path)
    result = run_search(instance)
    assert seen[0]["prompt"] == evidence_plan_prompt("BASE\n")
    assert seen[1]["prefix"] == "v6.6/evidence-continuation" and seen[1]["session_role"] == "A"
    assert seen[1]["invocation_id"] == "r.A.1.plan" and "read any existing test file" in seen[1]["prompt"]
    merged = json.loads(result.trajectory_path.read_text())
    assert len(merged["steps"]) == 3
    check = json.loads((tmp_path / "store/v6.6/evidence-plan-check.json").read_text())
    assert check["initial_missing"] == ["test_read"] and check["remaining_missing"] == []


def test_complete_search_runs_once_and_other_roles_are_untouched(tmp_path, monkeypatch):
    seen = []

    async def parent(self, **kwargs):
        seen.append(kwargs)
        return role_result(tmp_path, f"r{len(seen)}", ["stop"],
                           [("find", "/w/spec"), ("read", "/w/spec/tests/test_a.py")])

    monkeypatch.setattr(V62Controller, "_run_role", parent)
    instance = controller(tmp_path)
    run_search(instance)
    asyncio.run(instance._run_role(role=RoleName.V5_SUMMARY_REVIEWER, prompt="BASE\n", worktree="/w",
                                   prefix="v5/summary"))
    assert len(seen) == 2 and seen[1]["prompt"] == "BASE\n"


def test_a_search_that_ended_in_an_api_error_is_an_infrastructure_failure(tmp_path, monkeypatch):
    async def parent(self, **kwargs):
        return role_result(tmp_path, "failed", ["error", "error"], [("ls", "/w/spec")])

    monkeypatch.setattr(V62Controller, "_run_role", parent)
    with pytest.raises(PiApiFailure, match="A_INFRASTRUCTURE"):
        run_search(controller(tmp_path))


def test_config_versions():
    config = V66Config()
    assert (config.workflow_prompt_version, config.workflow_controller_version) == ("v6.6.4", "v6.6.4")
    assert config.checkpoint_max_attempts == 2 and config.checkpoint_max_per_target == 4
    assert config.checkpoint_batch_max_targets == 8
    assert config.solver_view_max_points == 12 and config.evidence_plan_continuations == 1


def response(content):
    return {"choices": [{"message": {"content": content, "reasoning_content": "r"},
                         "finish_reason": "stop"}], "usage": {"prompt_tokens": 1}}


def test_client_retries_once_with_feedback_writes_views_and_never_resamples(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    client = CheckpointClient(model="m", base_url="http://x", api_key="k", max_per_target=4,
                              max_attempts=2, timeout_sec=5)
    sent = []
    replies = [response("not json"), response(json.dumps(payload([point()])))]
    monkeypatch.setattr(client, "_post", lambda messages: sent.append(messages) or replies.pop(0))
    result = asyncio.run(client.generate(packet=copy.deepcopy(PACKET), ledger=LEDGER, artifacts=store))
    assert result["status"] == "passed" and len(result["attempts"]) == 2
    assert sent[1][-1]["role"] == "user" and "violates the contract" in sent[1][-1]["content"]
    assert (tmp_path / "v6.6/checkpoints/solver-view.md").read_text().startswith("- [analog] O001/D2")
    assert result["summary"]["checkpoints"] == 1 and result["summary"]["coverage_cells"] == 16
    with pytest.raises(RuntimeError, match="resampling"):
        asyncio.run(client.generate(packet=copy.deepcopy(PACKET), ledger=LEDGER, artifacts=store))


def test_client_reports_invalid_output_after_the_last_attempt(tmp_path, monkeypatch):
    client = CheckpointClient(model="m", base_url="http://x", api_key="k", max_per_target=4,
                              max_attempts=2, timeout_sec=5)
    monkeypatch.setattr(client, "_post", lambda messages: response("{}"))
    result = asyncio.run(client.generate(packet=copy.deepcopy(PACKET), ledger=LEDGER,
                                         artifacts=ArtifactStore(tmp_path)))
    assert result["status"] == "invalid_output"
    assert not (tmp_path / "v6.6/checkpoints/checkpoints.json").exists()


def test_runner_sources_exist_and_runtime_requires_rg_fd(tmp_path):
    spec = importlib.util.spec_from_file_location("v66a_runner", PROJECT / "scripts/deepswe_v66_a/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    for name in run.FROZEN_SOURCES:
        assert (PROJECT / name).is_file(), name
    assert "external_agents_eval/workflows/solve_review_repair/v66_checkpoints.py" in run.FROZEN_SOURCES
    archive = tmp_path / "pi-runtime-a-rgfd.tar.gz"
    archive.write_bytes(b"x")
    manifest = {"archive": str(archive), "archive_sha256": run.file_sha256(archive),
                "pi_version": "0.84.2", "node_version": "v22.19.0", "rg_fd_included": True}
    (tmp_path / "runtime.json").write_text(json.dumps(manifest))
    assert run.runtime_manifest(archive)["rg_fd_included"] is True
    (tmp_path / "runtime.json").write_text(json.dumps({**manifest, "rg_fd_included": False}))
    with pytest.raises(ValueError):
        run.runtime_manifest(archive)


def test_probe_resolved_points_rank_first_in_the_solver_view():
    points = [point("D4", "open"), point("D2"), point("D3", "analog", probe={"cp": "CP9", "verified": True})]
    views = delivery_views(payload(points), max_solver_points=2)
    assert views["solver_points"][0]["probe"]["cp"] == "CP9"
    assert [p["resolution"] for p in views["solver_points"]] == ["analog", "open"]


def test_c_schema_has_no_siblings_field_after_the_revert():
    from external_agents_eval.workflows.solve_review_repair.v66_prompts import CHECKPOINT_SCHEMA

    item = CHECKPOINT_SCHEMA["properties"]["checkpoints"]["items"]
    assert "siblings" not in item["properties"] and "siblings" not in item["required"]
    assert "Siblings (hard rule)" not in checkpoint_system_prompt(4)
