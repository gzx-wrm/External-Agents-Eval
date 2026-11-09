"""featurebench_v6.4_woG: adapters, ledger-only A, the no-gate controller path and the runner."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_v5_controller import RepairWorkspace, ScriptedRunner, _assessment, _spec

from external_agents_eval.agents.mini_v62 import MINI_SYSTEM
from external_agents_eval.agents.mini_v64 import REPAIR_SYSTEM_V64, REVIEW_SYSTEM_V64, SUMMARY_SYSTEM_V64
from external_agents_eval.benchmarks.featurebench import v64
from external_agents_eval.benchmarks.featurebench.agents import FeatureBenchV63Agent
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore
from external_agents_eval.workflows.solve_review_repair.models import RoleName, TaskInput
from external_agents_eval.workflows.solve_review_repair.v5_controller import V5Controller
from external_agents_eval.workflows.solve_review_repair.v5_prompts import I_PROBE_RULES_V56
from external_agents_eval.workflows.solve_review_repair.v64_config import V64Config

PROJECT = Path(__file__).resolve().parents[1]
ACCEPT = {"summary": "No repair is required.", "decisions": [], "repair_requests": [],
          "new_questions": []}
REPAIR = {
    "summary": "One repair is required.",
    "decisions": [{"finding_ref": "I/O001", "disposition": "repair",
                   "reason": "The required behavior is violated.", "supporting_refs": ["I/O001"],
                   "sources": [{"kind": "task", "field": "description", "quote": "Add behavior"}]}],
    "repair_requests": [{"finding_refs": ["I/O001"], "change": "Implement the behavior.",
                         "validation_record_refs": ["I/O001"], "preserve_refs": ["R/P001"]}],
    "new_questions": [],
}


def load_runner():
    spec = importlib.util.spec_from_file_location("featurebench_v64_run",
                                                  PROJECT / "scripts/featurebench_v64/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_controller(tmp_path, s_output, i_status="satisfied"):
    scripts = {
        RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001", status=i_status)],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        RoleName.V5_SUMMARY_REVIEWER: [s_output],
        RoleName.REPAIR: ["repair complete"],
    }
    runner = ScriptedRunner(tmp_path, scripts)
    controller = V5Controller(role_runner=runner, workspace=RepairWorkspace(),
                              artifacts=ArtifactStore(tmp_path / "artifacts"), config=v64.V64WoGConfig())
    try:
        result = asyncio.run(controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0", base_worktree="base", candidate_worktree="candidate",
            repair_worktree="repair", run_id="fb-v64-wog",
        ))
    finally:
        controller.close()
    return result, runner, tmp_path / "artifacts/v5"


# -- framework parity --------------------------------------------------------

def test_wog_config_differs_from_v64_only_in_the_gate_switch():
    full, wog = asdict(V64Config()), asdict(v64.V64WoGConfig())
    assert {k for k in full if full[k] != wog[k]} == {"build_gate_enabled"}
    assert wog["build_gate_enabled"] is False and wog["prompt_version"] == "v5-6"
    assert wog["controller_version"] == "v5-5" and wog["review_skill"] == "v6.4-review-note-files"


def test_controller_without_gate_keeps_v56_prompts(tmp_path):
    result, runner, artifacts = run_controller(tmp_path, ACCEPT)
    assert result.route == "accept"
    i_prompt = next(c for c in runner.call_details if c["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER)["prompt"]
    assert I_PROBE_RULES_V56 in i_prompt and "v6.4-review-note-files/SKILL.md" in i_prompt
    assert not (artifacts / "build-gate").exists()


def test_controller_without_gate_repairs_without_build_check(tmp_path):
    result, runner, _ = run_controller(tmp_path, REPAIR, i_status="violated")
    assert result.route == "repair" and result.p1_patch == b"p1"
    repair = next(c for c in runner.call_details if c["role"] == RoleName.REPAIR)["prompt"]
    # The fixed v5-6 goal line mentions a build check "when present"; no BUILD CHECK section follows.
    assert repair.startswith("ROLE: Repair\nGoal:") and "\nBUILD CHECK\n" not in repair


# -- FeatureBench review backend ---------------------------------------------

def test_review_runner_uses_the_featurebench_baseline():
    runner = object.__new__(v64.FeatureBenchV64ReviewRunner)
    review = runner._mini_system(role=RoleName.V5_IMPLEMENTATION_REVIEWER, tools=("bash",), schemas=None)
    repair = runner._mini_system(role=RoleName.REPAIR, tools=("bash",), schemas=None)
    summary = runner._mini_system(role=RoleName.V5_SUMMARY_REVIEWER, tools=("read",), schemas=None)
    assert review == REVIEW_SYSTEM_V64 and summary == SUMMARY_SYSTEM_V64
    assert repair == REPAIR_SYSTEM_V64.replace("{baseline}", "/testbed")
    assert "/testbed is the original baseline" in repair and "/app" not in repair
    corrector = runner._mini_system(role=RoleName.JSON_CORRECTOR, tools=(), schemas=None)
    assert corrector.startswith(MINI_SYSTEM) and "/app" not in corrector


def test_review_agent_has_no_gate_and_uses_testbed():
    with pytest.raises(ValueError, match="without the build gate"):
        v64.FeatureBenchV64ReviewAgent(logs_dir=None, build_gate=object())
    assert v64.FeatureBenchV64ReviewAgent.workdir == "/testbed"
    assert v64.FeatureBenchV64ReviewAgent.workflow_config_type is v64.V64WoGConfig


# -- ledger-only A -------------------------------------------------------------

class FakeController:
    def __init__(self, ledger):
        self.ledger, self.calls = ledger, 0
        self.ledger_quality, self.ledger_warnings, self.ledger_error = "valid", [], None

    async def prepare_ledger(self, task, **kwargs):
        self.calls += 1
        self.ledger_quality = "degraded" if self.ledger else "unavailable"
        self.ledger_warnings = ["anchor"] if self.ledger else []
        self.ledger_error = None if self.ledger else "schema"
        return self.ledger


@pytest.mark.parametrize("ledger", [{"obligations": [{"id": "O001"}]}, None])
def test_ledger_agent_stops_right_after_a(monkeypatch, ledger):
    fake = FakeController(ledger)
    monkeypatch.setattr(FeatureBenchV63Agent, "_make_controller", lambda self, **kwargs: fake)
    agent = object.__new__(v64.FeatureBenchLedgerAgent)
    controller = agent._make_controller()
    with pytest.raises(v64.LedgerReady) as stopped:
        asyncio.run(controller.prepare_ledger("task", base_worktree="spec-review"))
    assert fake.calls == 1 and stopped.value.ledger == ledger
    assert stopped.value.quality == ("degraded" if ledger else "unavailable")
    assert stopped.value.error == (None if ledger else "schema")


def test_probe_file_prefers_the_package_root():
    assert v64.pick_probe_file(["setup.py", "xarray/__init__.py"], "xarray") == "xarray/__init__.py"
    assert v64.pick_probe_file(["src/lightning/__init__.py", "a.py"], "lightning") == "src/lightning/__init__.py"
    assert v64.pick_probe_file(["README.md", "b/c.py", "a/z.py"], None) == "a/z.py"
    assert v64.pick_probe_file(["README.md"], None) == "README.md"
    assert v64.pick_probe_file([], None) is None


# -- runner ------------------------------------------------------------------

def test_solver_statement_carries_exactly_one_supplement():
    run = load_runner()
    text = run.augment_statement("## Task\nDo it at /testbed/x.py", {"obligations": []})
    assert text.startswith("## Task\nDo it at /testbed/x.py\n\n")
    assert text.count(run.HEADING) == 1 and text.count(run.GUIDANCE) == 1


def test_image_groups_and_library_names():
    run = load_runner()
    names = ["libercoders/featurebench-specs_transformers-instance_5c00e64e",
             "libercoders/featurebench-specs_pydantic-instance_78693714",
             "libercoders/featurebench-specs_pytorch_lightning-instance_83799db4",
             "libercoders/featurebench-specs_zzz-instance_1"]
    assert run.image_group(names[2]) == "pytorch_lightning"
    order = [run.image_group(n) for n in run.group_order(names)]
    assert order == ["pydantic", "pytorch_lightning", "transformers", "zzz"]
    assert run.library_name({"level": 2, "repo_settings": "{}"}) is None
    assert run.library_name({"level": 1, "repo_settings": json.dumps({"library_name": "astropy"})}) == "astropy"


def test_reused_checks_and_events(tmp_path):
    run = load_runner()
    assert run.reference_passed({"completed": True, "resolved": True, "f2p_total": 3})
    assert not run.reference_passed({"completed": True, "resolved": True, "f2p_total": 0})
    assert run.reference_passed({"cpu_reference_kind": "pristine_original_tests", "completed": True,
                                 "probe_passed": True, "tests_executed": 4, "test_exit_code": 0,
                                 "tests_failed": 0, "tests_errors": 0})
    assert run.control_passed({"patch_applied": True, "resolved": False})
    assert not run.control_passed({"patch_applied": True, "resolved": True})
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps(e) for e in [
        {"event": "V5_REVIEW_INTEGRITY_WARNING", "status": "restored"},
        {"event": "V6_FINAL_WORKTREE_NORMALIZED", "discarded_changes": True},
        {"event": "V6_FINAL_WORKTREE_NORMALIZED", "discarded_changes": False}]) + "\nnot json\n")
    assert run.count_events(events) == {"reviewer_worktree_restored": 1, "final_worktree_discarded": 1}
    (tmp_path / "solver").mkdir()
    run.set_aside(tmp_path / "solver")
    assert not (tmp_path / "solver").exists() and len(list(tmp_path.glob("solver.incomplete-*"))) == 1


def digest(task):
    return hashlib.sha256(json.dumps(task, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def fake_batch(root, tasks, *, cpu=None, iers=False, results):
    (root / "configs").mkdir(parents=True)
    (root / "configs/native-deepseek-v4.1-flash.yaml").write_text("agent: {}\n")
    if cpu is not None:
        (root / "configs/cpu-evaluation.json").write_text(json.dumps(cpu))
    if iers:
        (root / "assets").mkdir()
        (root / "assets/Leap_Second.dat").write_text("# leap data\n")
        (root / "configs/iers-leap-seconds.json").write_text(json.dumps(
            {"file": "../assets/Leap_Second.dat", "sha256": "x", "expires": "2027-06-28T00:00:00+00:00"}))
    (root / "manifests").mkdir()
    (root / "manifests/tasks-private.json").write_text(json.dumps(tasks))
    (root / "manifests/tasks-public.json").write_text(json.dumps(
        [{"instance_id": t["instance_id"], "private_task_sha256": digest(t)} for t in tasks]))
    for task in tasks:
        out = root / "outputs" / task["instance_id"]
        for rel, value in [("reference/summary.json", {"completed": True, "resolved": True, "f2p_total": 2}),
                           ("control/summary.json", {"patch_applied": True, "resolved": False}),
                           ("cpu-readiness.json", {"passed": True}),
                           ("generation/isolation.json", {"tree_sha": "tree-" + task["instance_id"]}),
                           ("result.json", results[task["instance_id"]])]:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(json.dumps(value))
        (out / "generation/model.patch").write_text("diff\n")


def make_task(tid, image, level=1):
    return {"instance_id": tid, "image_name": image, "level": level, "problem_statement": "## Task " + tid,
            "repo_settings": json.dumps({"library_name": tid.split("-")[0]})}


def test_prepare_binds_m0_batches_and_per_task_environments(tmp_path, monkeypatch):
    run = load_runner()
    scored = {"status": "scored", "reward": 1, "official": {"f2p_passed": 2, "f2p_total": 2,
                                                            "p2p_passed": 1, "p2p_total": 1, "resolved": True}}
    first, second = tmp_path / "m0-a", tmp_path / "m0-b"
    a = [make_task("pydantic-a", "x/featurebench-specs_pydantic-instance_1"),
         make_task("gpu-x", "x/featurebench-specs_gpu-instance_9")]
    b = [make_task("astropy-b", "x/featurebench-specs_astropy-instance_2", level=2)]
    fake_batch(first, a, results={"pydantic-a": scored, "gpu-x": {"status": "gpu_deferred", "reward": None}})
    fake_batch(second, b, cpu={"timeout_one": 60, "environment": {"HF_ENDPOINT": "https://hf-mirror.com"},
                               "network": "host"}, iers=True, results={"astropy-b": dict(scored, reward=0)})
    runtime = tmp_path / "native"
    runtime.mkdir()
    for name in run.NATIVE_SCRIPTS:
        (runtime / name).write_text("# " + name)
    cache = tmp_path / "image-cache.json"
    cache.write_text("{}")
    monkeypatch.setattr(run, "M0_RUNS", [str(first), str(second)])
    monkeypatch.setattr(run, "NATIVE_RUNTIME", str(runtime))
    monkeypatch.setattr(run, "IMAGE_CACHE_CONFIG", str(cache))
    monkeypatch.setattr(run, "EXCLUDED", {"gpu-x": "deferred"})
    monkeypatch.setattr(run, "TASK_COUNT", 2)
    root = tmp_path / "exp"
    selection = run.prepare(root)
    rows = {row["task_id"]: row for row in selection["tasks"]}
    assert [row["task_id"] for row in selection["tasks"]] == ["pydantic-a", "astropy-b"]
    assert rows["pydantic-a"]["scoring_environment"]["cpu_config_source"] is None
    assert run.task_cpu_config(root, "pydantic-a") is None
    assert json.loads(run.task_cpu_config(root, "astropy-b").read_text())["environment"]["HF_ENDPOINT"]
    iers = json.loads(run.task_iers_config(root, "astropy-b").read_text())
    assert iers["file"] == "Leap_Second.dat" and (run.scoring_config_dir(root, "astropy-b") / "Leap_Second.dat").exists()
    assert rows["astropy-b"]["library"] is None and rows["pydantic-a"]["library"] == "pydantic"
    assert rows["pydantic-a"]["m0_tree_sha"] == "tree-pydantic-a" and rows["astropy-b"]["m0"]["reward"] == 0
    assert selection["build_gate"] is False and selection["reference_control_reused_from_m0"] is True
    with pytest.raises(ValueError, match="new, empty"):
        run.prepare(root)
    monkeypatch.setattr(run, "EXCLUDED", {"gpu-x": "deferred", "pydantic-a": "wrongly excluded"})
    with pytest.raises(ValueError, match="excluded task has an M0 score"):
        run.prepare(tmp_path / "exp2")


# -- meson-python editable installs ---------------------------------------------

class RecordingContainer:
    def __init__(self):
        self.commands = []

    def exec_run(self, command, workdir=None, demux=False):
        from types import SimpleNamespace
        self.commands.append((command[-1], workdir))
        return SimpleNamespace(exit_code=0, output=(b"", b""))


def test_v64_environment_skips_editable_redirect_only_in_worktrees():
    container = RecordingContainer()
    bridge = v64.FeatureBenchV64Environment(container, instance_id="task")
    bridge.exec_sync("python -c 1", cwd="/tmp/external_eval/worktrees/repair")
    assert "MESONPY_EDITABLE_SKIP" not in container.commands[-1][0]
    bridge.set_editable({"builds": [{"marker": "MESONPY_EDITABLE_SKIP", "build_path": "/testbed/build/cp311"}]})
    bridge.exec_sync("python -c 1", cwd="/tmp/external_eval/worktrees/repair")
    command, workdir = container.commands[-1]
    assert "export MESONPY_EDITABLE_SKIP=/testbed/build/cp311\n" in command and workdir.endswith("repair")
    assert "conda activate testbed" in command and "PYTHONPATH=/tmp/external_eval/worktrees/repair:" in command
    for directory in ("/testbed", "/testbed/pandas", None):
        bridge.exec_sync("python -c 1", cwd=directory)
        assert "MESONPY_EDITABLE_SKIP" not in container.commands[-1][0]


def load_editable_helper():
    spec = importlib.util.spec_from_file_location(
        "featurebench_editable_assets", PROJECT / "external_agents_eval/benchmarks/featurebench/editable_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(cwd, *args):
    import subprocess
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout


def test_editable_build_outputs_are_seeded_and_kept_out_of_diffs(tmp_path, monkeypatch):
    helper = load_editable_helper()
    repo = tmp_path / "testbed"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/__init__.py").write_text("from pkg._version_meson import v\n")
    (repo / ".gitignore").write_text("*.so\nbuild\n")
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "add", "-A")
    git(repo, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base")
    build = repo / "build/cp311"
    (build / "meson-info").mkdir(parents=True)
    (build / "pkg").mkdir()
    (build / "_version_meson.py").write_text("v = '1'\n")
    (build / "pkg/_x.cpython-311-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    (build / "meson-info/intro-install_plan.json").write_text(json.dumps({
        "targets": {str(build / "pkg/_x.cpython-311-x86_64-linux-gnu.so"):
                    {"destination": "{py_platlib}/pkg/_x.cpython-311-x86_64-linux-gnu.so"}},
        "python": {str(build / "_version_meson.py"): {"destination": "{py_platlib}/pkg/_version_meson.py"},
                   str(repo / "pkg/__init__.py"): {"destination": "{py_platlib}/pkg/__init__.py"}},
    }))
    site = tmp_path / "site"
    site.mkdir()
    (site / "_pkg_editable_loader.py").write_text(
        "class MesonpyMetaFinder: pass\nMARKER = 'MESONPY_EDITABLE_SKIP'\n"
        f"install(\n    {{'pkg'}},\n    {str(build)!r},\n    ['true'],\n    True,\n)\n")
    monkeypatch.setattr(helper.sysconfig, "get_paths", lambda: {"purelib": str(site), "platlib": str(site)})
    monkeypatch.setattr(helper, "STATE", str(tmp_path / "state/scan.json"))
    summary = helper.scan(str(repo), rebuild=True)
    assert summary["builds"] == [{"names": ["pkg"], "build_path": str(build),
                                  "marker": "MESONPY_EDITABLE_SKIP", "targets": 2}]
    worktree = tmp_path / "worktrees/repair"
    git(repo, "worktree", "add", "-q", "--detach", str(worktree), "HEAD")
    report = helper.seed(str(worktree))
    assert report["files"] == 2 and report["excluded_locally"] == ["pkg/_version_meson.py"]
    assert (worktree / "pkg/_version_meson.py").read_text() == "v = '1'\n"
    assert (worktree / "pkg/_x.cpython-311-x86_64-linux-gnu.so").exists()
    assert git(worktree, "status", "--porcelain", "--untracked-files=all") == ""
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    helper.seed(str(worktree))  # idempotent: no duplicate exclude lines
    exclude = (repo / ".git/info/exclude").read_text()
    assert exclude.count("/pkg/_version_meson.py") == 1
    # A model edit to a tracked file is still visible.
    (worktree / "pkg/__init__.py").write_text("changed\n")
    assert "pkg/__init__.py" in git(worktree, "status", "--porcelain")
