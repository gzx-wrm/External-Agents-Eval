import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.benchmarks.deepswe import pier_compat, runner
from external_agents_eval.benchmarks.deepswe.config import (
    CODEX_CONFIG_FILE,
    CODEX_VERSION,
)
from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest
from external_agents_eval.benchmarks.deepswe.pier_compat import (
    DEEPSWE_BUILDER_ENV,
    _is_transient_environment_start_error,
    classify_verifier_failure,
    install_custom_endpoint_egress_patch,
    install_docker_builder_patch,
    install_docker_image_cleanup_patch,
    install_environment_start_retry_patch,
    install_registry_start_retry_patch,
    install_resource_isolation_patch,
    install_verifier_observability_patch,
)
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    DEVELOPMENT_TASK_IDS,
    ReviewerExperimentManifest,
    load_frozen_few_shots,
)
from external_agents_eval.benchmarks.deepswe.runner import (
    PIER_COMPAT_MODULE,
    _apply_docker_cleanup_compat,
    _docker_builder_name,
    _resolve_pier_executable,
    _run_pier_process,
    build_pier_command,
)


def _manifest(tmp_path):
    return DeepSWEManifest(
        path=tmp_path / "manifest.json",
        task_ids=("task-b", "task-a"),
        sample_seed=7,
        sample_size=2,
        population_count=113,
        population_fingerprint="a" * 64,
        raw={},
    )


def _reviewer_manifest(tmp_path):
    return ReviewerExperimentManifest(
        path=tmp_path / "p0-manifest.json",
        source_job_id="source-job",
        source_job_dir=tmp_path / "source-job",
        source_manifest_path=tmp_path / "manifest.json",
        source_manifest_sha256="a" * 64,
        task_ids=("task-a",),
        patches={},
        raw={},
    )


def _write_few_shots(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "mode": "public_only",
                "development_task_ids": list(DEVELOPMENT_TASK_IDS),
            }
        )
    )
    for task_id in DEVELOPMENT_TASK_IDS:
        task_dir = root / task_id
        task_dir.mkdir(parents=True)
        for filename in (
            "stage_a_input.json",
            "stage_a_output.json",
            "stage_b_input_summary.json",
            "stage_b_output.json",
        ):
            (task_dir / filename).write_text(json.dumps({"task_id": task_id}))


def test_pier_command_uses_exact_manifest_and_secret_template(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi",
        jobs_dir=Path("/jobs"),
        job_name="fixed-pi",
    )
    assert "external_agents_eval.agents.pier.pi:PiPierAgent" in command
    assert "--agent" not in command
    assert "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" in command
    assert command.count("--include-task-name") == 2
    assert command[-4:] == [
        "--include-task-name",
        "task-b",
        "--include-task-name",
        "task-a",
    ]
    assert "--sample-seed" not in command


def test_v6_pier_command_passes_fork_plan(tmp_path):
    plan = tmp_path / "fork.json"
    command = build_pier_command(
        pier_executable="pier", checkout=Path("/checkout"),
        manifest=_manifest(tmp_path), agent_id="pi-v6-review-repair",
        jobs_dir=tmp_path, job_name="fork-test", fork_plan=plan,
        include_task_ids=("task-a",),
    )
    assert f"fork_plan={plan.resolve()}" in command


def test_v6_fork_plans_run_as_one_concurrent_pier_job(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    mapping = {}
    for task_id in ("task-a", "task-b"):
        source = tmp_path / task_id / "source"
        source.mkdir(parents=True)
        plan = tmp_path / task_id / "fork.json"
        plan.write_text(json.dumps({
            "task_id": task_id, "start_at": "Solver", "source_run": "source",
        }), encoding="utf-8")
        mapping[task_id] = f"{task_id}/fork.json"
    map_path = tmp_path / "plans.json"
    map_path.write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(runner, "validate_manifest_against_checkout", lambda _manifest, _checkout: checkout)
    monkeypatch.setattr(runner, "_resolve_pier_executable", lambda command: command)
    created = []
    removed = []
    commands = []
    monkeypatch.setattr(runner, "_create_dedicated_builder", lambda name: created.append(name))
    monkeypatch.setattr(runner, "_prewarm_egress_proxy", lambda _name: 1)
    monkeypatch.setattr(runner, "_remove_dedicated_builder", lambda name: removed.append(name))
    monkeypatch.setattr(runner, "_run_pier_process", lambda command, _env: commands.append(command))

    runner.run_pier(
        checkout=checkout, manifest=_manifest(tmp_path), agent_id="pi-v6-review-repair",
        jobs_dir=tmp_path / "jobs", job_name="batch", fork_plans=map_path,
        num_workers=2, pier_executable="/bin/true",
    )

    record = json.loads((tmp_path / "jobs/batch.run.json").read_text())
    assert len(commands) == len(created) == len(removed) == 1
    assert created == removed
    assert record["task_ids"] == ["task-a", "task-b"]
    assert record["fork_start_at_by_task"] == {"task-a": "Solver", "task-b": "Solver"}
    assert f"fork_plans={map_path.resolve()}" in commands[0]
    assert commands[0][commands[0].index("--n-concurrent") + 1] == "2"
    assert commands[0].count("--include-task-name") == 2


def test_v6_fork_plans_reject_incomplete_task_selection(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    for task_id in ("task-a", "task-b"):
        plan = tmp_path / f"{task_id}.json"
        plan.write_text(json.dumps({"task_id": task_id, "start_at": "A"}), encoding="utf-8")
    mapping = tmp_path / "plans.json"
    mapping.write_text(json.dumps({
        "task-a": "task-a.json", "task-b": "task-b.json",
    }), encoding="utf-8")
    monkeypatch.setattr(runner, "validate_manifest_against_checkout", lambda _manifest, _checkout: checkout)
    with pytest.raises(ValueError, match="must match the selected tasks"):
        runner.run_pier(
            checkout=checkout, manifest=_manifest(tmp_path), agent_id="pi-v6-review-repair",
            jobs_dir=tmp_path / "jobs", fork_plans=mapping, task_ids=("task-a",), dry_run=True,
        )


def test_pier_command_can_select_one_manifest_task(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi",
        jobs_dir=Path("/jobs"),
        job_name="subset",
        include_task_ids=("task-a",),
    )
    assert command.count("--include-task-name") == 1
    assert command[-2:] == ["--include-task-name", "task-a"]


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ((), "empty"),
        (("task-a", "task-a"), "unique"),
        (("unknown",), "outside"),
    ],
)
def test_pier_command_rejects_invalid_task_subsets(tmp_path, selected, message):
    with pytest.raises(ValueError, match=message):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi",
            jobs_dir=Path("/jobs"),
            job_name="subset",
            include_task_ids=selected,
        )


def test_review_command_freezes_digest_and_forbids_task_level_retries(tmp_path):
    digest = "b" * 64
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=Path("/jobs"),
        job_name="review",
        few_shot_dir=Path("/frozen/few-shots"),
        few_shot_digest=digest,
    )
    assert f"few_shot_digest={digest}" in command
    assert "few_shot_dir=/frozen/few-shots" in command
    assert "review_arm=E0" in command
    timeout_index = command.index("--agent-timeout-multiplier")
    assert command[timeout_index + 1] == "4"
    with pytest.raises(ValueError, match="forbids automatic task-level retries"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi-review-repair",
            jobs_dir=Path("/jobs"),
            job_name="review",
            attempts=2,
        )


def test_v5_command_accepts_frozen_p0_without_legacy_contract_arguments(tmp_path):
    reviewer_manifest = _reviewer_manifest(tmp_path)
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi-v5-review-repair",
        jobs_dir=Path("/jobs"),
        job_name="review-v5",
        reviewer_manifest=reviewer_manifest,
        workflow_timeout_sec=3600,
    )

    assert (
        "external_agents_eval.agents.pier.pi_v5_solve_review_repair:"
        "PiV5SolveReviewRepairPierAgent"
    ) in command
    assert f"p0_manifest={reviewer_manifest.path.resolve()}" in command
    assert "workflow_timeout_sec=3600" in command
    assert not any("review_contract_version=" in argument for argument in command)
    assert not any("review_prompt_version=" in argument for argument in command)
    assert command[-2:] == ["--include-task-name", "task-a"]

    with pytest.raises(ValueError, match="forbids automatic task-level retries"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi-v5-review-repair",
            jobs_dir=Path("/jobs"),
            job_name="review-v5-retry",
            attempts=2,
        )

    with pytest.raises(ValueError, match="valid only for pi-review-repair"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi-v5-review-repair",
            jobs_dir=Path("/jobs"),
            job_name="review-v5-legacy-contract",
            review_contract_version="3.0",
        )


def test_v6_command_generates_p0_and_rejects_frozen_p0(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi-v6-review-repair",
        jobs_dir=Path("/jobs"),
        job_name="review-v6",
        workflow_timeout_sec=3600,
    )

    assert (
        "external_agents_eval.agents.pier.pi_v6_solve_review_repair:"
        "PiV6SolveReviewRepairPierAgent"
    ) in command
    assert "workflow_timeout_sec=3600" in command
    assert not any(argument.startswith("p0_manifest=") for argument in command)
    assert "--agent-timeout-multiplier" in command

    with pytest.raises(ValueError, match="valid only for a Pi reviewer workflow"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi-v6-review-repair",
            jobs_dir=Path("/jobs"),
            job_name="review-v6-frozen",
            reviewer_manifest=_reviewer_manifest(tmp_path),
        )

    with pytest.raises(ValueError, match="forbids automatic task-level retries"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi-v6-review-repair",
            jobs_dir=Path("/jobs"),
            job_name="review-v6-retry",
            attempts=2,
        )


def test_v5_dry_run_records_v5_workflow_without_few_shots(
    tmp_path, monkeypatch, capsys
):
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="pi-v5-review-repair",
        jobs_dir=tmp_path / "jobs",
        environment="modal",
        dry_run=True,
        workflow_timeout_sec=3600,
    )

    metadata = json.loads(capsys.readouterr().out)
    assert metadata["workflow"] == "solve-review-repair-v5"
    assert metadata["controller_version"] == "v5-4"
    assert metadata["prompt_version"] == "v5-5"
    assert metadata["p0_manifest"] is None
    assert "workflow_timeout_sec=3600" in metadata["command"]
    assert not any(
        "few_shot" in argument or "review_contract_version" in argument
        for argument in metadata["command"]
    )


def test_v6_dry_run_records_pre_solver_ledger_workflow(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="pi-v6-review-repair",
        jobs_dir=tmp_path / "jobs",
        environment="modal",
        dry_run=True,
        workflow_timeout_sec=3600,
    )

    metadata = json.loads(capsys.readouterr().out)
    assert metadata["workflow"] == "solve-review-repair-v6.1"
    assert metadata["review_protocol"] == "v5"
    assert metadata["review_prompt_version"] == "v5-5"
    assert metadata["review_controller_version"] == "v5-4"
    assert metadata["ledger_timing"] == "before-solver"
    assert metadata["p0_manifest"] is None
    assert "workflow_timeout_sec=3600" in metadata["command"]

def test_review_command_can_select_v2_source_and_repair_audit_arms(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=Path("/jobs"),
        job_name="review-v2",
        few_shot_dir=Path("/frozen/few-shots-v2"),
        few_shot_digest="b" * 64,
        review_arm="E2",
    )
    assert "review_arm=E2" in command

    with pytest.raises(ValueError, match="valid only"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi",
            jobs_dir=Path("/jobs"),
            job_name="invalid-v2",
            review_arm="E2",
        )


def test_review_command_can_select_v3_without_changing_arm_meaning(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=Path("/jobs"),
        job_name="review-v3",
        few_shot_dir=Path("/frozen/few-shots-v3"),
        few_shot_digest="b" * 64,
        review_arm="E2",
        review_contract_version="3.0",
    )
    assert "review_arm=E2" in command
    assert "review_contract_version=3.0" in command

    with pytest.raises(ValueError, match="incompatible"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi-review-repair",
            jobs_dir=Path("/jobs"),
            job_name="invalid-v3",
            review_arm="E0",
            review_contract_version="3.0",
        )


def test_review_command_can_select_v4_prompt_with_v3_contract(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=Path("/jobs"),
        job_name="review-v4",
        few_shot_dir=Path("/frozen/few-shots-v4"),
        few_shot_digest="b" * 64,
        review_arm="E2",
        review_contract_version="3.0",
        review_prompt_version="4.0",
    )
    assert "review_contract_version=3.0" in command
    assert "review_prompt_version=4.0" in command

    with pytest.raises(ValueError, match="requires contract v3.0"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi-review-repair",
            jobs_dir=Path("/jobs"),
            job_name="invalid-v4",
            review_arm="E2",
            review_contract_version="2.0",
            review_prompt_version="4.0",
        )

    with pytest.raises(ValueError, match="valid only"):
        build_pier_command(
            pier_executable="pier",
            checkout=Path("/checkout"),
            manifest=_manifest(tmp_path),
            agent_id="pi",
            jobs_dir=Path("/jobs"),
            job_name="invalid-v4-agent",
            review_prompt_version="4.0",
        )


def test_v2_review_run_records_contract_and_uses_synthetic_few_shots(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    source = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v2"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )
    monkeypatch.setattr(runner, "_run_pier_process", lambda _command, _env: None)
    monkeypatch.setattr(runner, "_resolve_pier_executable", lambda command: command)

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=tmp_path / "jobs",
        job_name="v2-review",
        environment="modal",
        pier_executable="/bin/true",
        few_shot_dir=source,
        review_arm="E2",
    )

    record = json.loads((tmp_path / "jobs/v2-review.run.json").read_text())
    assert record["workflow"] == "solve-review-repair-once-v2"
    assert record["experiment_arm"] == "E2"
    assert record["contract_version"] == "2.0"
    assert record["few_shot_mode"] == "synthetic_public_contracts_v2"
    assert "review_arm=E2" in record["command"]


def test_v3_review_run_records_obligation_workflow_and_few_shots(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    source = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v3"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )
    monkeypatch.setattr(runner, "_run_pier_process", lambda _command, _env: None)
    monkeypatch.setattr(runner, "_resolve_pier_executable", lambda command: command)

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=tmp_path / "jobs",
        job_name="v3-review",
        environment="modal",
        pier_executable="/bin/true",
        few_shot_dir=source,
        review_arm="E2",
        review_contract_version="3.0",
    )

    record = json.loads((tmp_path / "jobs/v3-review.run.json").read_text())
    assert record["workflow"] == "solve-review-repair-once-v3"
    assert record["experiment_arm"] == "E2"
    assert record["contract_version"] == "3.0"
    assert record["few_shot_mode"] == "synthetic_public_contracts_v3"
    assert "review_contract_version=3.0" in record["command"]


def test_v4_review_run_records_prompt_version_and_matching_few_shots(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    source = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v4"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )
    monkeypatch.setattr(runner, "_run_pier_process", lambda _command, _env: None)
    monkeypatch.setattr(runner, "_resolve_pier_executable", lambda command: command)

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=tmp_path / "jobs",
        job_name="v4-review",
        environment="modal",
        pier_executable="/bin/true",
        few_shot_dir=source,
        review_arm="E2",
        review_contract_version="3.0",
        review_prompt_version="4.0",
    )

    record = json.loads((tmp_path / "jobs/v4-review.run.json").read_text())
    assert record["workflow"] == "solve-review-repair-once-v3"
    assert record["contract_version"] == "3.0"
    assert record["prompt_version"] == "4.0"
    assert "review_prompt_version=4.0" in record["command"]


def test_review_run_rejects_prompt_and_few_shot_version_mismatch(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    v3 = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v3"
    v4 = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v4"
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )

    common = {
        "checkout": checkout,
        "manifest": _manifest(tmp_path),
        "agent_id": "pi-review-repair",
        "jobs_dir": tmp_path / "jobs",
        "environment": "modal",
        "pier_executable": "/bin/true",
        "dry_run": True,
        "review_arm": "E2",
        "review_contract_version": "3.0",
    }
    with pytest.raises(ValueError, match="prompt version does not match"):
        runner.run_pier(
            **common,
            few_shot_dir=v3,
            review_prompt_version="4.0",
        )
    with pytest.raises(ValueError, match="prompt version does not match"):
        runner.run_pier(**common, few_shot_dir=v4)


def test_review_run_snapshots_few_shots_and_records_batch_digest(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    source = tmp_path / "few-shots"
    _write_few_shots(source)
    expected = load_frozen_few_shots(source)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )
    monkeypatch.setattr(runner, "_run_pier_process", lambda _command, _env: None)
    monkeypatch.setattr(runner, "_resolve_pier_executable", lambda command: command)

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="pi-review-repair",
        jobs_dir=tmp_path / "jobs",
        job_name="frozen-review",
        environment="modal",
        pier_executable="/bin/true",
        few_shot_dir=source,
    )

    record = json.loads((tmp_path / "jobs/frozen-review.run.json").read_text())
    snapshot = Path(record["few_shot_dir"])
    assert snapshot != source
    assert snapshot.parent.name == ".review_inputs"
    assert record["few_shot_digest"] == expected.digest
    assert record["few_shot_file_sha256"] == dict(expected.file_sha256)
    assert f"few_shot_digest={expected.digest}" in record["command"]
    assert record["agent_timeout_multiplier"] == 4.0
    assert record["internal_role_timeouts_sec"] == {
        "solver": 10_800,
        "repair": 10_800,
        "specification_reviewer": 3_600,
        "patch_reviewer": 3_600,
        "json_corrector": 1_800,
    }

    expected.stage_a[0].write_text(
        json.dumps({"task_id": DEVELOPMENT_TASK_IDS[0], "changed": True})
    )
    assert load_frozen_few_shots(snapshot).digest == expected.digest


def test_run_record_contains_only_selected_tasks(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )
    monkeypatch.setattr(runner, "_run_pier_process", lambda _command, _env: None)
    monkeypatch.setattr(runner, "_resolve_pier_executable", lambda command: command)

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="pi",
        jobs_dir=tmp_path / "jobs",
        job_name="selected-run",
        environment="modal",
        pier_executable="/bin/true",
        task_ids=("task-a",),
    )

    record = json.loads((tmp_path / "jobs/selected-run.run.json").read_text())
    assert record["task_ids"] == ["task-a"]
    assert record["command"].count("--include-task-name") == 1


def test_codex_command_uses_pier_native_agent_and_secret_safe_config(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="codex",
        jobs_dir=Path("/jobs"),
        job_name="fixed-codex",
    )

    agent_index = command.index("--agent")
    assert command[agent_index + 1] == "codex"
    assert "--agent-import-path" not in command
    assert f"version={CODEX_VERSION}" in command
    assert f"config_toml_file={CODEX_CONFIG_FILE.resolve()}" in command
    assert "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" in command

    config = CODEX_CONFIG_FILE.read_text(encoding="utf-8")
    assert 'web_search = "disabled"' in config
    assert 'base_url = "https://api.deepseek.com/"' in config
    assert 'wire_api = "responses"' in config
    assert 'env_key = "DEEPSEEK_API_KEY"' in config
    assert "experimental_bearer_token" not in config


def test_codex_dry_run_records_pier_and_provider_metadata(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, _checkout: checkout,
    )

    runner.run_pier(
        checkout=checkout,
        manifest=_manifest(tmp_path),
        agent_id="codex",
        jobs_dir=tmp_path / "jobs",
        environment="modal",
        dry_run=True,
    )

    metadata = json.loads(capsys.readouterr().out)
    assert metadata["agent"] == "codex"
    assert metadata["agent_version"] == CODEX_VERSION
    assert metadata["agent_implementation"] == "pier-native"
    assert metadata["model_provider"] == "deepseek"
    assert metadata["model_provider_base_url"] == "https://api.deepseek.com/"
    assert metadata["model_provider_wire_api"] == "responses"
    assert "agent_upstream_commit" not in metadata


def test_hermes_command_carries_turn_limit(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="hermes",
        jobs_dir=Path("/jobs"),
        job_name="fixed-hermes",
        max_turns=12,
    )
    index = command.index("--agent-kwarg")
    assert command[index + 1] == "max_turns=12"


def test_hermes_command_omits_turn_limit_by_default(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=Path("/checkout"),
        manifest=_manifest(tmp_path),
        agent_id="hermes",
        jobs_dir=Path("/jobs"),
        job_name="fixed-hermes",
    )
    assert "--agent-kwarg" not in command
    assert not any(argument.startswith("max_turns=") for argument in command)


def test_pier_resolves_next_to_active_python_when_not_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.touch()
    pier = bin_dir / "pier"
    pier.touch(mode=0o755)
    monkeypatch.setattr(runner.sys, "executable", str(python))
    monkeypatch.setattr(runner.shutil, "which", lambda _command: None)

    assert _resolve_pier_executable("pier") == str(pier)


def test_docker_command_uses_cleanup_compat_wrapper():
    command = _apply_docker_cleanup_compat(["/venv/bin/pier", "run", "--yes"], "docker")
    assert command == [runner.sys.executable, "-m", PIER_COMPAT_MODULE, "run", "--yes"]
    assert _apply_docker_cleanup_compat(command, "modal") is command


def test_cleanup_patch_keeps_final_image_removal_pass():
    class FakeTrial:
        _external_eval_cleanup_patch = False

        def __init__(self):
            self._is_agent_environment_stopped = False
            self.calls = []

        async def _stop_agent_environment(self, *, keep_images=False):
            if self._is_agent_environment_stopped:
                return
            self.calls.append(keep_images)
            self._is_agent_environment_stopped = True

    install_docker_image_cleanup_patch(FakeTrial)
    trial = FakeTrial()
    asyncio.run(trial._stop_agent_environment(keep_images=True))
    assert trial._is_agent_environment_stopped is False
    asyncio.run(trial._stop_agent_environment())
    assert trial.calls == [True, False]
    assert trial._is_agent_environment_stopped is True


def test_cleanup_patch_retains_builder_cache_between_trials(monkeypatch):
    class FakeTrial:
        _external_eval_cleanup_patch = False

        def __init__(self):
            self._is_agent_environment_stopped = False

        async def _stop_agent_environment(self, *, keep_images=False):
            self._is_agent_environment_stopped = True

    monkeypatch.setenv(DEEPSWE_BUILDER_ENV, "dedicated-builder")
    install_docker_image_cleanup_patch(FakeTrial)
    trial = FakeTrial()

    asyncio.run(trial._stop_agent_environment())

    assert trial._is_agent_environment_stopped is True


def test_environment_retry_recognizes_proxy_apt_transport_failures_only():
    assert _is_transient_environment_start_error(
        "target pier-egress-proxy: Failed to fetch http://ports.ubuntu.com/pkg.deb "
        "Unable to connect to ports.ubuntu.com:80"
    )
    assert not _is_transient_environment_start_error(
        "target pier-egress-proxy: E: Unable to locate package squid-typo"
    )
    assert not _is_transient_environment_start_error(
        "main: Failed to fetch a project dependency: package not found"
    )


def test_registry_start_retries_two_transient_token_failures(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    class FakeTrialExecution:
        _external_eval_registry_retry_patch = False

        def __init__(self):
            self.calls = 0
            self.environment = SimpleNamespace(logger=SimpleNamespace(warning=lambda *_args: None))

        async def start_environment(self, *, force_build):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError(
                    "failed to fetch oauth token: Post https://auth.docker.io/token: EOF"
                )

    monkeypatch.setattr(pier_compat.asyncio, "sleep", fake_sleep)
    install_registry_start_retry_patch(FakeTrialExecution)
    execution = FakeTrialExecution()

    asyncio.run(execution.start_environment(force_build=False))

    assert execution.calls == 3
    assert sleeps == [1, 2]


def test_environment_start_records_retry_count(tmp_path, monkeypatch):
    async def no_wait(_seconds):
        return None

    class FakeTrialExecution:
        _external_eval_environment_retry_patch = False

        def __init__(self):
            self.calls = 0
            self.environment = SimpleNamespace(
                logger=SimpleNamespace(warning=lambda *_args: None),
                trial_paths=SimpleNamespace(trial_dir=tmp_path),
            )

        async def start_environment(self, *, force_build):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError(
                    "pier-egress-proxy: Failed to fetch package; Unable to connect"
                )

    monkeypatch.setattr(pier_compat.asyncio, "sleep", no_wait)
    install_environment_start_retry_patch(FakeTrialExecution)
    execution = FakeTrialExecution()

    asyncio.run(execution.start_environment(force_build=False))

    record = json.loads((tmp_path / "environment_start.json").read_text())
    assert record == {
        "schema_version": "1.0",
        "status": "succeeded",
        "attempts": 3,
        "max_attempts": 3,
    }


def test_registry_start_does_not_retry_unrelated_agent_or_runtime_error(monkeypatch):
    async def fail_if_called(_seconds):
        raise AssertionError("sleep should not be called")

    class FakeTrialExecution:
        _external_eval_registry_retry_patch = False

        def __init__(self):
            self.calls = 0
            self.environment = SimpleNamespace(logger=SimpleNamespace(warning=lambda *_args: None))

        async def start_environment(self, *, force_build):
            self.calls += 1
            raise RuntimeError("repository setup failed")

    monkeypatch.setattr(pier_compat.asyncio, "sleep", fail_if_called)
    install_registry_start_retry_patch(FakeTrialExecution)
    execution = FakeTrialExecution()

    with pytest.raises(RuntimeError, match="repository setup failed"):
        asyncio.run(execution.start_environment(force_build=False))
    assert execution.calls == 1


def test_registry_start_stops_after_three_transient_failures(monkeypatch):
    async def no_wait(_seconds):
        return None

    class FakeTrialExecution:
        _external_eval_registry_retry_patch = False

        def __init__(self):
            self.calls = 0
            self.environment = SimpleNamespace(logger=SimpleNamespace(warning=lambda *_args: None))

        async def start_environment(self, *, force_build):
            self.calls += 1
            raise RuntimeError("registry-1.docker.io: TLS handshake timeout")

    monkeypatch.setattr(pier_compat.asyncio, "sleep", no_wait)
    install_registry_start_retry_patch(FakeTrialExecution)
    execution = FakeTrialExecution()

    with pytest.raises(RuntimeError, match="TLS handshake timeout"):
        asyncio.run(execution.start_environment(force_build=True))
    assert execution.calls == 3


def test_verifier_observability_classifies_cancelled_attempt_as_timeout(tmp_path):
    class FakeTrial:
        _external_eval_verifier_observability_patch = False

        def __init__(self):
            self.trial_paths = SimpleNamespace(trial_dir=tmp_path)

        async def _verify_once(self, **_kwargs):
            raise asyncio.CancelledError()

    install_verifier_observability_patch(FakeTrial)
    trial = FakeTrial()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(trial._verify_once())

    attempts = json.loads((tmp_path / "verifier_attempts.json").read_text())
    summary = json.loads((tmp_path / "verifier_observability.json").read_text())
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["failure_class"] == "verifier_timeout"
    assert summary["original_failure_class"] == "verifier_timeout"
    assert summary["retry_count"] == 0


def test_resource_isolation_patch_binds_agent_and_separate_verifier(tmp_path, monkeypatch):
    class FakeTrial:
        _external_eval_resource_isolation_patch = False

        def __init__(self, config):
            self.config = config
            self._environment = live_environment
            self.trial_paths = SimpleNamespace(trial_dir=tmp_path)

    task_env = SimpleNamespace(cpus=None, memory_mb=None)
    live_environment = SimpleNamespace(
        task_env_config=task_env,
        _override_cpus=None,
        _override_memory_mb=None,
    )
    config = SimpleNamespace(
        environment=SimpleNamespace(override_cpus=None, override_memory_mb=None)
    )
    monkeypatch.setenv("EXTERNAL_EVAL_TRIAL_CPUS", "2")
    monkeypatch.setenv("EXTERNAL_EVAL_TRIAL_MEMORY_MB", "768")
    install_resource_isolation_patch(FakeTrial)
    trial = FakeTrial(config)

    assert trial.config.environment.override_cpus == 2
    assert trial.config.environment.override_memory_mb == 768
    assert trial._environment._override_cpus == 2
    assert trial._environment._override_memory_mb == 768
    assert task_env.cpus == 2
    assert task_env.memory_mb == 768
    record = json.loads((tmp_path / "resource_limits.json").read_text())
    assert record["cpus"] == 2
    assert record["memory_mb"] == 768


def test_verifier_failure_classifier_keeps_test_failures_separate():
    assert classify_verifier_failure("pytest assertion failed") == "test_failure"
    assert classify_verifier_failure("docker build failed") == "verifier_build_failure"
    assert classify_verifier_failure("connection reset by peer") == "verifier_infrastructure_flake"


def test_custom_endpoint_egress_patch_allows_literal_ip_on_port_3000(monkeypatch):
    from pier.environments import agent_setup
    from pier.environments.docker import docker as docker_environment

    original = agent_setup.squid_bootstrap_command
    monkeypatch.setattr(agent_setup, "squid_bootstrap_command", original)
    install_custom_endpoint_egress_patch()

    script = agent_setup.squid_bootstrap_command()
    assert "acl Safe_ports port 80 443 3000" in script
    assert 'acl allowed_ips dst "/tmp/allowed_ips.txt"' in script
    assert 'acl allowed_ip_urls url_regex -i "/tmp/allowed_ip_urls.txt"' in script
    assert "http_access allow authenticated allowed_ips" in script
    assert "http_access allow authenticated allowed_ip_urls" in script
    assert (
        docker_environment.write_docker_proxy_compose
        is agent_setup.write_docker_proxy_compose
    )
    assert docker_environment.write_agent_dockerfile is agent_setup.write_agent_dockerfile


def test_agent_apt_transport_patch_updates_json_run_command(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    command = [
        "/bin/bash",
        "-c",
        "set -eu\n"
        "if command -v apt-get >/dev/null 2>&1; then\n"
        "  apt-get update\n"
        "  DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates\n"
        "  rm -rf /var/lib/apt/lists/*\n"
        "fi",
    ]
    dockerfile.write_text("FROM example\nRUN " + json.dumps(command) + "\n")

    pier_compat._patch_agent_apt_transport(dockerfile)

    patched = json.loads(dockerfile.read_text().splitlines()[1].removeprefix("RUN "))[-1]
    assert "s#http://#https://#g" in patched
    assert 'Acquire::https::Verify-Peer "false";' in patched
    assert "rm -f /etc/apt/apt.conf.d/99-external-eval-bootstrap" in patched


def test_builder_patch_adds_explicit_compose_builder(monkeypatch):
    class FakeDockerEnvironment:
        _external_eval_builder_patch = False

        @staticmethod
        def _build_command(*, force_build):
            return ["build", "--no-cache"] if force_build else ["build"]

    monkeypatch.setenv(DEEPSWE_BUILDER_ENV, "dedicated-builder")
    install_docker_builder_patch(FakeDockerEnvironment)

    assert FakeDockerEnvironment._build_command(force_build=False) == [
        "build",
        "--builder",
        "dedicated-builder",
    ]
    assert FakeDockerEnvironment._build_command(force_build=True) == [
        "build",
        "--no-cache",
        "--builder",
        "dedicated-builder",
    ]


def test_builder_name_is_stable_and_job_scoped(tmp_path):
    first = _docker_builder_name(tmp_path, "job-a")
    assert first == _docker_builder_name(tmp_path, "job-a")
    assert first != _docker_builder_name(tmp_path, "job-b")
    assert first.startswith("external-eval-deepswe-")


def test_builder_inherits_container_reachable_docker_desktop_proxy(monkeypatch):
    docker_info = {
        "HttpProxy": "http.docker.internal:3128",
        "HttpsProxy": "http.docker.internal:3128",
        "NoProxy": "hubproxy.docker.internal",
    }

    def fake_run(command, **_kwargs):
        assert command == ["docker", "info", "--format", "{{json .}}"]
        return runner.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps(docker_info),
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._docker_desktop_proxy_driver_options() == [
        "--driver-opt",
        "env.http_proxy=http://http.docker.internal:3128",
        "--driver-opt",
        "env.https_proxy=http://http.docker.internal:3128",
        "--driver-opt",
        "env.no_proxy=hubproxy.docker.internal",
    ]


def test_builder_does_not_forward_unreachable_or_credentialed_proxy(monkeypatch):
    docker_info = {
        "HttpProxy": "http://127.0.0.1:7890",
        "HttpsProxy": "http://user:secret@http.docker.internal:3128",
        "NoProxy": "localhost",
    }
    result = runner.subprocess.CompletedProcess(
        ["docker", "info"],
        returncode=0,
        stdout=json.dumps(docker_info),
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: result)

    assert runner._docker_desktop_proxy_driver_options() == []


def test_builder_creation_adds_detected_proxy_options(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return_code = 1 if command[:3] == ["docker", "buildx", "inspect"] else 0
        return runner.subprocess.CompletedProcess(command, returncode=return_code)

    proxy_options = [
        "--driver-opt",
        "env.https_proxy=http://http.docker.internal:3128",
    ]
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "_docker_desktop_proxy_driver_options",
        lambda: proxy_options,
    )

    runner._create_dedicated_builder("dedicated-builder")

    assert calls[-1] == [
        "docker",
        "buildx",
        "create",
        "--name",
        "dedicated-builder",
        "--driver",
        "docker-container",
        *proxy_options,
        "--bootstrap",
    ]


def test_proxy_prewarm_retries_and_uses_pier_generated_context(monkeypatch):
    calls = []
    sleeps = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        context = Path(command[-1])
        dockerfile = (context / "Dockerfile").read_text()
        start_script = (context / "start-squid.sh").read_text()
        assert "FROM ubuntu:24.04" in dockerfile
        assert "s#http://#https://#g" in dockerfile
        assert dockerfile.index("find /etc/apt") < dockerfile.index("apt-get update")
        assert 'Acquire::https::Verify-Peer "false";' in dockerfile
        assert "rm -f /etc/apt/apt.conf.d/99-external-eval-bootstrap" in dockerfile
        assert "apache2-utils ca-certificates squid" in dockerfile
        assert "exec squid -N" in start_script
        return runner.subprocess.CompletedProcess(
            command,
            returncode=1 if len(calls) < 3 else 0,
            stdout=(
                "target pier-egress-proxy: Failed to fetch package; Unable to connect"
                if len(calls) < 3
                else "built"
            ),
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    attempts = runner._prewarm_egress_proxy("dedicated-builder")

    assert attempts == 3
    assert sleeps == [1, 2]
    assert all(call[3:5] == ["--builder", "dedicated-builder"] for call in calls)


def test_reviewer_base_preflight_accepts_short_sha_and_rejects_mismatch(tmp_path):
    task_dir = tmp_path / "tasks/task-a/environment"
    task_dir.mkdir(parents=True)
    (task_dir / "Dockerfile").write_text("ARG BASE_SHA=7cef35b" + "a" * 33 + "\n")
    manifest = _reviewer_manifest(tmp_path)
    manifest.patches["task-a"] = SimpleNamespace(base_commit="7cef35b")

    runner._validate_reviewer_task_bases(
        checkout=tmp_path,
        manifest=manifest,
        task_ids=("task-a",),
    )

    manifest.patches["task-a"] = SimpleNamespace(base_commit="7cef35c")
    with pytest.raises(ValueError, match="does not match the active task Dockerfile"):
        runner._validate_reviewer_task_bases(
            checkout=tmp_path,
            manifest=manifest,
            task_ids=("task-a",),
        )


def test_docker_run_prewarms_once_and_retains_cache_until_job_end(tmp_path, monkeypatch):
    events = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "validate_manifest_against_checkout",
        lambda _manifest, checkout: checkout,
    )
    monkeypatch.setattr(runner, "_resolve_pier_executable", lambda command: command)
    monkeypatch.setattr(
        runner,
        "_create_dedicated_builder",
        lambda name: events.append(("create", name)),
    )
    monkeypatch.setattr(
        runner,
        "_prewarm_egress_proxy",
        lambda name: events.append(("prewarm", name)) or 2,
    )
    monkeypatch.setattr(
        runner,
        "_run_pier_process",
        lambda _command, _env: events.append(("run", None)),
    )
    monkeypatch.setattr(
        runner,
        "_remove_dedicated_builder",
        lambda name: events.append(("remove", name)),
    )

    runner.run_pier(
        checkout=tmp_path / "checkout",
        manifest=_manifest(tmp_path),
        agent_id="pi",
        jobs_dir=tmp_path / "jobs",
        job_name="cached-docker-run",
        pier_executable="pier",
        environment="docker",
    )

    record = json.loads((tmp_path / "jobs/cached-docker-run.run.json").read_text())
    builder = record["docker_builder"]
    assert events == [
        ("create", builder),
        ("prewarm", builder),
        ("run", None),
        ("remove", builder),
    ]
    assert record["docker_builder_cache_policy"] == (
        "retain-through-job-remove-after-job"
    )
    assert record["docker_egress_proxy_prewarm"] == {
        "status": "completed",
        "attempts": 2,
        "max_attempts": 3,
    }


def test_pier_process_waits_for_cleanup_after_keyboard_interrupt(monkeypatch):
    class FakeProcess:
        def __init__(self):
            self.wait_calls = 0
            self.terminated = False

        def wait(self):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            return 130

        def terminate(self):
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)

    try:
        _run_pier_process(["pier", "run"], {})
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("KeyboardInterrupt should be re-raised after child cleanup")

    assert process.wait_calls == 2
    assert process.terminated is False
