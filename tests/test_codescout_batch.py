from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[1] / "scripts/deepswe_codescout/batch.py"
SPEC = importlib.util.spec_from_file_location("codescout_batch_tests", PATH)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def worker(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    write(root / "configs/codescout.json", asdict(batch.Config()))
    task = {"task_id": "case", "task_checksum": "checksum", "instruction": "original issue",
            "image": "original-tag", "cpus": 2, "memory_mb": 8192}
    record = {"task_id": "case", "m0_image": {"id": "sha256:original",
              "repo_digests": ["repository@sha256:pinned"]}, "preprocessing_reused": False}
    calls, states = [], []
    client = SimpleNamespace(close=lambda: None, images=SimpleNamespace(remove=lambda *a, **k: None))

    def get_image(client, task, output):
        calls.append("image")
        assert task["image"] == "repository@sha256:pinned"
        return SimpleNamespace(id="sha256:original"), False

    def preprocess(command, **kwargs):
        calls.append("preprocess")
        assert command[-1] == "case" and kwargs["timeout"] == 3600
        output = root / "outputs/case"
        write(output / "augmented-task.json", {**task, "instruction": "enhanced issue"})
        write(output / "preprocessing/codescout/result.json", {
            "targets": 5, "analyzed": 4, "retained": 3, "model_calls_in_protocol": 6,
            "prompt_tokens": 600, "completion_tokens": 60, "new_prompt_tokens": 600,
            "new_completion_tokens": 60, "new_request_attempts": 6,
        })

    def generate(augmented, root, output, config, image, mode):
        calls.append("solve")
        assert augmented["instruction"] == "enhanced issue" and mode == "generate"
        assert config == {"frozen_native_config": True}
        output.mkdir(parents=True, exist_ok=True)
        info = {"exit_status": "Submitted"}
        write(output / "agent-info.json", info)
        (output / "model.patch").write_text("generated patch")
        write(output / "native.traj.json", {"messages": [{"role": "assistant", "extra": {
            "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                   "prompt_tokens_details": {"cached_tokens": 30}}}}}]})
        return "generated patch", info

    def grade(root, original, image_id, patch, output):
        calls.append("grade")
        assert original["instruction"] == "original issue"
        assert image_id == "sha256:original" and patch == "generated patch"
        return {"status": "scored", "task_id": "case", "reward": 0}

    native = SimpleNamespace(
        docker=SimpleNamespace(from_env=lambda **k: client, errors=SimpleNamespace(APIError=RuntimeError)),
        get_image=get_image, generate=generate, grade=grade,
    )
    monkeypatch.setattr(batch.subprocess, "run", preprocess)
    update = lambda tid, phase, **values: states.append((phase, values))
    return root, task, record, native, calls, states, update


def execute(worker):
    root, task, record, native, _calls, _states, update = worker
    return batch.run_one(root, record, task, native, {"frozen_native_config": True}, update)


def test_end_to_end_order_and_normal_zero_reward(worker):
    result = execute(worker)
    assert worker[4] == ["image", "preprocess", "solve", "grade"]
    assert result["status"] == "scored" and result["reward"] == 0
    assert result["native_agent_completed"]
    assert result["solver_usage"] == {"prompt_tokens": 100, "completion_tokens": 10,
                                      "cached_prompt_tokens": 30}
    assert result["preprocessing"]["prompt_tokens"] == 600
    assert worker[5][-1][0] == "scored"
    # A valid reward=0 is terminal, not a reason to sample again.
    execute(worker)
    assert worker[4] == ["image", "preprocess", "solve", "grade"]


def test_verifier_recovery_reuses_solver_patch_and_conversation(worker):
    original = worker[3].grade

    def unavailable(*args, **kwargs):
        raise RuntimeError("Scorer unavailable")

    worker[3].grade = unavailable
    first = execute(worker)
    assert first["reward"] is None and first["failed_phase"] == "official_evaluation"
    worker[3].grade = original
    second = execute(worker)
    assert second["reward"] == 0
    assert worker[4].count("solve") == 1 and worker[4].count("preprocess") == 2


def test_interrupted_solver_is_not_automatically_resampled(worker):
    def interrupted(*args, **kwargs):
        worker[4].append("solve")
        raise RuntimeError("Interrupted Solver")

    worker[3].generate = interrupted
    assert execute(worker)["failed_phase"] == "solving"
    second = execute(worker)
    assert second["reward"] is None and "do not automatically resample" in second["error"]
    assert worker[4].count("solve") == 1 and "grade" not in worker[4]


def test_preprocessing_failure_never_scores_unmodified_baseline(worker, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("preprocess", 3600)

    monkeypatch.setattr(batch.subprocess, "run", timeout)
    result = execute(worker)
    assert result["failed_phase"] == "codescout_preprocessing" and result["reward"] is None
    assert "solve" not in worker[4] and "grade" not in worker[4]


def test_wrong_image_stops_before_any_model_or_grading(worker):
    worker[3].get_image = lambda *args: (SimpleNamespace(id="sha256:other"), False)
    result = execute(worker)
    assert result["failed_phase"] == "image_preparation" and result["reward"] is None
    assert worker[4] == []


def test_docker_failure_is_a_recorded_infrastructure_outcome(worker):
    def unavailable(**kwargs):
        raise RuntimeError("Docker unavailable")

    worker[3].docker.from_env = unavailable
    result = execute(worker)
    assert result["failed_phase"] == "image_preparation" and result["reward"] is None


@pytest.fixture
def sources(tmp_path, monkeypatch):
    parent = tmp_path / "m0"
    write(parent / "preflight/readiness.json", {"passed": True})
    tasks = [{"task_id": f"task-{i:02d}", "task_checksum": f"checksum-{i}",
              "instruction": f"original issue {i}", "image": f"image-{i}"} for i in range(30)]
    roots = []
    for number, selected in enumerate([tasks[:8], tasks[8:]]):
        root = tmp_path / f"source-{number}"
        (root / "scripts").mkdir(parents=True)
        (root / "grader").mkdir()
        for name in ["m0_run.py", "native_compat.py", "sanitize_repository.py"]:
            (root / "scripts" / name).write_text("# frozen native implementation\n")
        for name in ["native-deepseek-v4.1-flash.yaml", "upstream-mini.yaml"]:
            write(root / "configs" / name, {"unchanged_native_configuration": True})
        write(root / "manifests/public-tasks.json", tasks)
        write(root / "manifests/eval30_current.json", {"task_ids": [task["task_id"] for task in tasks]})
        write(root / "selection.json", {"m0_root": str(parent), "tasks": [
            {"task_id": task["task_id"], "task_checksum": task["task_checksum"],
             "m0_image": {"id": "sha256:" + task["task_id"],
                          "repo_digests": ["repo@sha256:" + task["task_id"]]}} for task in selected]})
        roots.append(root)
    native = SimpleNamespace(load_tasks=lambda root: batch.read(root / "manifests/public-tasks.json"))
    monkeypatch.setattr(batch, "load_native", lambda root: native)
    return roots


def test_prepare_freezes_exact30_and_smoke_subset(sources, tmp_path):
    full = batch.prepare(tmp_path / "full", sources)
    assert full["task_count"] == 30
    assert full["ft_ledger_or_review_included"] is False
    assert all(not record["preprocessing_reused"] for record in full["tasks"])
    smoke = batch.prepare(tmp_path / "smoke", sources, ["task-03"])
    assert smoke["task_count"] == 1 and smoke["tasks"][0]["task_id"] == "task-03"
    assert batch.validate(tmp_path / "smoke")[0] == smoke
    with pytest.raises(ValueError, match="outside"):
        batch.prepare(tmp_path / "outside", sources, ["unknown"])


def test_prepare_rejects_configuration_drift(sources, tmp_path):
    (sources[1] / "configs/native-deepseek-v4.1-flash.yaml").write_text("different configuration")
    with pytest.raises(ValueError, match="differs"):
        batch.prepare(tmp_path / "invalid", sources)


def test_launch_validation_rejects_mutated_frozen_inputs(sources, tmp_path):
    root = tmp_path / "experiment"
    batch.prepare(root, sources)
    (root / "configs/codescout.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        batch.validate(root)


def test_utf8_recovery_selects_only_unsampled_header_failures(tmp_path):
    path = PATH.with_name("finish_index_failures.py")
    spec = importlib.util.spec_from_file_location("codescout_index_continuation", path)
    continuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(continuation)
    records = [{"task_id": "utf8"}, {"task_id": "other"}, {"task_id": "solved"}]
    for tid in ["utf8", "other"]:
        write(tmp_path / "outputs" / tid / "result.json", {
            "reward": None, "failed_phase": "codescout_preprocessing"})
        log = tmp_path / "logs" / (tid + ".preprocessing.log")
        log.parent.mkdir(exist_ok=True)
        log.write_text("UnicodeDecodeError header.decode unexpected end of data" if tid == "utf8"
                       else "Network failure")
    write(tmp_path / "outputs/solved/result.json", {"reward": 0, "status": "scored"})
    assert continuation.eligible_cases(tmp_path, {"tasks": records}) == ["utf8"]
    write(tmp_path / "outputs/utf8/preprocessing/codescout/scope/request.json", {"attempt": 1})
    with pytest.raises(ValueError, match="already reached the model"):
        continuation.eligible_cases(tmp_path, {"tasks": records})
