from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents.mini_ft_review import (
    FrozenMiniReviewAgent,
    FrozenMiniReviewRunner,
    normalization_command,
    write_frozen_fork,
)
from external_agents_eval.agents.mini_v63 import REVIEW_SYSTEM
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes, sha256_json
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v6_fork import V6ForkPlan, V6StageSource

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ft_review_runner", PROJECT / "scripts/deepswe_ft_m1_review/run.py",
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


@pytest.mark.parametrize("image_changes", [False, True])
def test_conversion_preserves_candidate_tree_and_real_index(tmp_path, image_changes):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    (repo / "module.py").write_text("initial = 1\n")
    (repo / "removed.txt").write_text("original\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "baseline")
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    if image_changes:
        (repo / "module.py").write_text("image_default = 2\n")
        (repo / "environment.txt").write_text("image runtime\n")
    git(repo, "add", "-A")
    baseline = git(repo, "diff", "--cached", "--binary", "--full-index", base)
    canonical_baseline = git(repo, "diff", "--cached", "--binary", "--no-ext-diff", base, "--", ".")
    baseline_file = tmp_path / "baseline.patch"
    baseline_file.write_bytes(baseline)
    (repo / "module.py").write_text("feature = 3\n")
    (repo / "removed.txt").unlink()
    (repo / "new.bin").write_bytes(b"\x00binary\xff")
    (repo / "link").symlink_to("module.py")
    git(repo, "add", "-A")
    candidate = git(repo, "diff", "--cached", "--binary", "--full-index", base)
    candidate_file = tmp_path / "p0.patch"
    candidate_file.write_bytes(candidate)
    expected_tree = git(repo, "write-tree").decode().strip()
    original_index = (repo / ".git/index").read_bytes()
    delta = tmp_path / "delta.patch"
    command = normalization_command(base, str(baseline_file), str(candidate_file), str(delta))
    process = subprocess.run(["bash", "-c", command], cwd=repo, text=True,
                             capture_output=True, check=True)
    receipt = json.loads(process.stdout)
    assert delta.with_name(delta.name + ".baseline").read_bytes() == canonical_baseline
    assert receipt["candidate_tree"] == receipt["reconstructed_tree"]
    assert receipt["candidate_tree"] == expected_tree
    assert receipt["candidate_unchanged"]
    assert (repo / ".git/index").read_bytes() == original_index
    assert candidate_file.read_bytes() == candidate
    # Match V6 capture_patch's exact textual representation, not only behavior.
    expected = git(repo, "diff", "--cached", "--binary", "--no-ext-diff",
                   receipt["baseline_tree"], "--", ".")
    assert delta.read_bytes() == expected
    assert b"new.bin" in expected and b"removed.txt" in expected and b"link" in expected


def test_stage_reuse_binds_exact_ledger_patch_and_environment(tmp_path):
    ledger = {"summary": "Original", "obligations": [],
              "preservation_obligations": [], "open_questions": []}
    path = write_frozen_fork(
        tmp_path, task_id="case", base_commit="base", baseline_patch=b"image",
        ledger=ledger, candidate_patch=b"candidate", provenance={"m1_output": "original-m1"},
    )
    plan = V6ForkPlan.load(path)
    assert plan.start_at == "I/R" and set(plan.sources) == {"A", "Solver"}
    source = V6StageSource(plan.source("Solver"), task_id="case", base_commit="base",
                           baseline_sha256=sha256_bytes(b"image"))
    assert source.patch("Solver", ledger_sha256=sha256_json(ledger)) == b"candidate"
    assert json.loads(plan.source("A").read_text()) == ledger
    with pytest.raises(ValueError, match="another A ledger"):
        source.patch("Solver", ledger_sha256="changed")
    with pytest.raises(ValueError, match="Frozen review input changed"):
        write_frozen_fork(tmp_path, task_id="case", base_commit="base", baseline_patch=b"image",
                          ledger=ledger, candidate_patch=b"new sample",
                          provenance={"m1_output": "original-m1"})


@pytest.mark.parametrize("role", [RoleName.SOLVER, RoleName.V5_SPECIFICATION_REVIEWER])
def test_review_extension_cannot_resample_upstream(role):
    runner = object.__new__(FrozenMiniReviewRunner)
    with pytest.raises(ValueError, match="resampling is forbidden"):
        asyncio.run(runner.run_role(role=role, prompt="", worktree="/app", tools=(),
                                    timeout_sec=30, artifact_prefix="forbidden"))


def test_deepswe_changes_only_the_baseline_path_in_mini_role_guidance():
    runner = object.__new__(FrozenMiniReviewRunner)
    assert runner._mini_system(role=RoleName.V5_IMPLEMENTATION_REVIEWER, tools=("bash",),
                               schemas=None) == REVIEW_SYSTEM
    system = runner._mini_system(role=RoleName.REPAIR, tools=("bash",), schemas=None)
    assert "/testbed" not in system and "/app is the original baseline" in system
    assert FrozenMiniReviewAgent.workdir == "/app"


def test_prepare_freezes_completed_source_and_rejects_later_changes(tmp_path, monkeypatch):
    source, base = tmp_path / "m1", tmp_path / "m0"
    (base / "grader").mkdir(parents=True)
    batch.save(base / "preflight/readiness.json", {"passed": True})
    (source / "scripts").mkdir(parents=True)
    for name in ["m0_run.py", "native_compat.py", "sanitize_repository.py"]:
        (source / "scripts" / name).write_text("# frozen native source\n")
    task = {"task_id": "case", "task_checksum": "checksum", "instruction": "Public issue"}
    native = SimpleNamespace(load_tasks=lambda root: [task])
    monkeypatch.setattr(batch, "load_native", lambda root: native)
    patch = b"frozen native patch"
    ledger = {"summary": "Original", "obligations": [],
              "preservation_obligations": [], "open_questions": []}
    batch.save(source / "ledger.json", ledger)
    batch.save(source / "configs/native-deepseek-v4.1-flash.yaml", {})
    image = {"id": "image", "repo_digests": ["original@sha256:image"]}
    row = {"task_id": "case", "task_checksum": "checksum", "ledger_file": "ledger.json",
           "ledger_file_sha256": sha256_bytes((source / "ledger.json").read_bytes()),
           "ledger_sha256": sha256_json(ledger), "m0_image": image}
    batch.save(source / "selection.json", {"experiment_id": "ft_m1", "m0_root": str(base),
                                           "tasks": [row]})
    out = source / "outputs/case"
    batch.save(out / "result.json", {"status": "scored", "reward": 0,
                                     "patch_sha256": sha256_bytes(patch)})
    batch.save(out / "input-audit.json", {"ledger_sha256": sha256_json(ledger),
                                          "evidence_packet_included": False, "ledger_modified": False})
    batch.save(out / "prediction.json", {"exit_status": "Submitted", "model_patch": patch.decode()})
    batch.save(out / "image.json", image)
    batch.save(out / "isolation.json", {"network": "none"})
    (out / "model.patch").write_bytes(patch)
    (out / "environment-baseline.patch").write_bytes(b"baseline")
    root = tmp_path / "review"
    selection = batch.prepare(root, [source], {"case"})
    assert selection["ledger_and_p0_reused"] and not selection["new_evidence_packet"]
    assert (root / "grader").resolve() == (base / "grader").resolve()
    assert batch.validate_selection(root, native) == selection
    with pytest.raises(ValueError, match="new, empty"):
        batch.prepare(root, [source], {"case"})
    (out / "model.patch").write_bytes(b"a different sample")
    with pytest.raises(ValueError, match="Frozen source changed"):
        batch.validate_selection(root, native)


@pytest.mark.parametrize("route", ["accept", "hold", "repair", "preflight", "incomplete"])
def test_review_then_grade_uses_final_patch_and_never_sends_score_to_agent(
    tmp_path, monkeypatch, route,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-host-only-key")
    source_root = tmp_path / "m1"
    source = source_root / "outputs/case"
    source.mkdir(parents=True)
    source.joinpath("model.patch").write_bytes(b"original full P0")
    source.joinpath("environment-baseline.patch").write_bytes(b"image changes")
    ledger = {"summary": "Frozen", "obligations": [],
              "preservation_obligations": [], "open_questions": []}
    batch.save(source_root / "ledger.json", ledger)
    batch.save(source_root / "configs/native-deepseek-v4.1-flash.yaml", {
        "environment": {"run_args": [], "forward_env": []},
    })
    root = tmp_path / "review"
    row = {"task_id": "case", "task": {"task_id": "case", "cpus": 2, "memory_mb": 8192,
            "instruction": "Original public issue"}, "m1_output": str(source),
           "source_root": str(source_root), "ledger_file": str(source_root / "ledger.json"),
           "ledger_sha256": sha256_json(ledger), "p0_sha256": sha256_bytes(b"original full P0"),
           "image": {"id": "image-id", "repo_digests": ["frozen@sha256:image"]}, "m1_reward": 0}
    container = SimpleNamespace(removed=False)
    container.remove = lambda **kwargs: setattr(container, "removed", True)
    client = SimpleNamespace(containers=SimpleNamespace(get=lambda _: container), close=lambda: None)
    native_env = SimpleNamespace(container_id="container")
    events = []
    final = "repaired full P1" if route == "repair" else "original full P0"

    def checked(env, command):
        if command == "export final":
            return final
        return json.dumps({"candidate_unchanged": True, "candidate_tree": "tree",
                           "reconstructed_tree": "tree"})

    def prepare_repository(root, env, container, task, output):
        output.joinpath("environment-baseline.patch").write_bytes(b"image changes")
        return {"synthetic_commit": "base"}

    def grade(root, task, image_id, patch, output):
        assert container.removed  # The generation environment ends before official test injection.
        assert events == ["review"] and patch == final
        events.append("grade")
        return {"task_id": "case", "status": "scored", "reward": 1}

    class Bridge:
        def __init__(self, *args, **kwargs):
            pass

        async def download_file(self, remote, local):
            local.write_bytes(b"canonical image changes" if remote.endswith(".baseline")
                              else b"normalized P0")

    class Agent:
        def __init__(self, *, logs_dir, fork_plan, **kwargs):
            assert route != "preflight", "Offline preflight must not construct a model agent"
            assert kwargs["extra_env"] == {"DEEPSEEK_API_KEY": "synthetic-host-only-key"}
            self.logs_dir = logs_dir
            plan = V6ForkPlan.load(fork_plan)
            assert json.loads(plan.source("A").read_text()) == ledger
            assert (plan.source("Solver") / "outputs/v6-p0.patch").read_bytes() == b"normalized P0"
            source = V6StageSource(plan.source("Solver"), task_id="case", base_commit="base",
                                   baseline_sha256=sha256_bytes(b"canonical image changes"))
            assert source.patch("Solver", ledger_sha256=sha256_json(ledger)) == b"normalized P0"

        async def run(self, instruction, environment, context):
            assert instruction == "Original public issue"
            assert "reward" not in instruction and "official" not in instruction
            events.append("review")
            batch.save(self.logs_dir.parent / "v6_review_repair/manifest.json", {
                "workflow_state": "safe_fallback" if route == "incomplete" else "completed",
                "route": route, "repair_executed": route == "repair",
            })

    native = SimpleNamespace(
        docker=SimpleNamespace(from_env=lambda **kw: client),
        get_image=lambda *a: (SimpleNamespace(id="image-id"), False),
        get_environment=lambda opts: native_env, prepare_repository=prepare_repository,
        checked=checked, upload=lambda *a: None, snapshot_command=lambda base: "export final", grade=grade,
    )
    monkeypatch.setattr(batch, "IsolatedEnvironment", Bridge)
    monkeypatch.setattr(batch, "FrozenMiniReviewAgent", Agent)
    if route == "incomplete":
        with pytest.raises(RuntimeError, match="no fallback result is scored"):
            batch.review_case(native, root, row)
        assert events == ["review"] and container.removed
        return
    result = batch.review_case(native, root, row, preflight=route == "preflight")
    if route == "preflight":
        assert events == [] and container.removed
        assert result["passed"] and result["model_api_calls"] == 0
        return
    assert events == ["review", "grade"]
    assert result["route"] == route and result["p0_reward"] == 0
    assert result["p1_sha256"] == sha256_bytes(final.encode())
    assert source.joinpath("model.patch").read_bytes() == b"original full P0"
    assert "synthetic-host-only-key" not in json.dumps(batch.read(root / "outputs/case/review-result.json"))
