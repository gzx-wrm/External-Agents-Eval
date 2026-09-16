import json
import shutil
from pathlib import Path

import pytest

from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    DEVELOPMENT_TASK_IDS,
    EVALUATION_TASK_IDS,
    build_reviewer_experiment_manifest,
    freeze_few_shots,
    git_object_ids_equivalent,
    load_frozen_few_shots,
    load_reviewer_experiment_manifest,
)
from external_agents_eval.workflows.solve_review_repair.config import DEFAULT_SCHEMA_DIR
from external_agents_eval.workflows.solve_review_repair.validators import ContractValidator


def write_few_shots(root):
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


def test_frozen_experiment_has_two_disjoint_development_and_sixteen_evaluation_tasks():
    assert len(DEVELOPMENT_TASK_IDS) == 2
    assert len(EVALUATION_TASK_IDS) == 16
    assert set(DEVELOPMENT_TASK_IDS).isdisjoint(EVALUATION_TASK_IDS)


def test_git_object_id_equivalence_accepts_safe_abbreviations_only():
    assert git_object_ids_equivalent("7cef35b", "7cef35b" + "a" * 33)
    assert git_object_ids_equivalent("A" * 7, "a" * 40)
    assert not git_object_ids_equivalent("7cef35c", "7cef35b" + "a" * 33)
    assert not git_object_ids_equivalent("abcdef", "abcdef" + "0" * 34)
    assert not git_object_ids_equivalent("not-a-sha", "a" * 40)


def test_load_frozen_few_shots_requires_all_eight_json_artifacts(tmp_path):
    write_few_shots(tmp_path)
    frozen = load_frozen_few_shots(tmp_path)
    assert len(frozen.stage_a) == 4
    assert len(frozen.stage_b) == 4
    assert len(frozen.file_sha256) == 9
    assert len(frozen.digest) == 64
    frozen.stage_b[-1].unlink()
    with pytest.raises(FileNotFoundError, match="stage_b_output"):
        load_frozen_few_shots(tmp_path)


def test_few_shot_preflight_rejects_hidden_evidence_markers(tmp_path):
    write_few_shots(tmp_path)
    contaminated = tmp_path / DEVELOPMENT_TASK_IDS[0] / "stage_b_output.json"
    contaminated.write_text(
        json.dumps({"task_id": DEVELOPMENT_TASK_IDS[0], "evidence": "verifier report"})
    )
    with pytest.raises(ValueError, match="forbidden marker"):
        load_frozen_few_shots(tmp_path)


def test_reference_assisted_provenance_requires_explicit_authorization(tmp_path):
    write_few_shots(tmp_path)
    sources = {
        task_id: {
            "instruction": "a" * 64,
            "p0": "b" * 64,
            "solution": "c" * 64,
            "test_patch": "d" * 64,
        }
        for task_id in DEVELOPMENT_TASK_IDS
    }
    provenance = {
        "schema_version": "1.0",
        "mode": "reference_assisted_development_only",
        "development_task_ids": list(DEVELOPMENT_TASK_IDS),
        "user_authorized": False,
        "source_sha256": sources,
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="explicit user authorization"):
        load_frozen_few_shots(tmp_path)

    provenance["user_authorized"] = True
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    assert load_frozen_few_shots(tmp_path).mode == "reference_assisted_development_only"


def test_few_shots_are_copied_to_content_addressed_read_only_snapshot(tmp_path):
    source = tmp_path / "source"
    write_few_shots(source)
    loaded = load_frozen_few_shots(source)
    snapshot = freeze_few_shots(loaded, tmp_path / "snapshots" / loaded.digest)

    assert snapshot.digest == loaded.digest
    assert snapshot.file_sha256 == loaded.file_sha256
    assert all(path.stat().st_mode & 0o222 == 0 for path in (*snapshot.stage_a, *snapshot.stage_b))

    loaded.stage_a[0].write_text(json.dumps({"task_id": DEVELOPMENT_TASK_IDS[0], "x": 1}))
    assert load_frozen_few_shots(snapshot.root).digest == snapshot.digest


def test_source_grounded_v2_few_shots_have_audited_provenance_and_valid_contracts():
    root = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v2"
    frozen = load_frozen_few_shots(root)
    provenance = json.loads((root / "provenance.json").read_text())

    assert frozen.contract_version == "2.0"
    assert set(frozen.case_ids).isdisjoint(DEVELOPMENT_TASK_IDS)
    assert set(frozen.case_ids).isdisjoint(EVALUATION_TASK_IDS)
    assert provenance["synthetic_only"] is True
    assert provenance["uses_benchmark_artifacts"] is False
    assert set(provenance["stage_relationships"]) == set(frozen.case_ids)

    validator = ContractValidator(DEFAULT_SCHEMA_DIR, contract_version="2.0")
    assert len(frozen.case_ids) == 2
    for case_id in frozen.case_ids:
        case = root / case_id
        stage_a_input = json.loads((case / "stage_a_input.json").read_text())
        ledger = json.loads((case / "stage_a_output.json").read_text())
        assessment = json.loads((case / "stage_b_output.json").read_text())
        audit = json.loads((case / "repair_audit_output.json").read_text())
        validator.validate_schema("build_input", stage_a_input)
        validator.validate_ledger(
            ledger,
            task_id=case_id,
            source_policy_sha256=ledger["source_policy_sha256"],
        )
        validator.validate_issue_source_anchors(
            ledger["sources"],
            title=stage_a_input["issue"]["title"],
            body=stage_a_input["issue"]["body"],
        )
        validator.validate_assessment(
            assessment,
            task_id=case_id,
            patch_sha256=assessment["patch_sha256"],
            ledger_sha256=assessment["ledger_sha256"],
            ledger=ledger,
        )
        effective = validator.build_effective_review_set(
            task_id=case_id,
            ledger_sha256=assessment["ledger_sha256"],
            assessment_sha256=audit["assessment_sha256"],
            ledger=ledger,
            assessment=assessment,
        )
        validator.validate_repair_audit(
            audit,
            task_id=case_id,
            p0_sha256=assessment["patch_sha256"],
            assessment_sha256=audit["assessment_sha256"],
            effective_review_set_sha256=audit["effective_review_set_sha256"],
            ledger=ledger,
            assessment=assessment,
            effective_review_set=effective,
            actual_changed_paths=set(audit["changed_paths"]),
        )

    independent = json.loads(
        (root / "synthetic-all-public-types" / "stage_b_input_summary.json").read_text()
    )
    assert "Independent counterfactual" in independent["example_scope"]

    assessments = [
        json.loads((root / case_id / "stage_b_output.json").read_text())
        for case_id in frozen.case_ids
    ]
    records = [
        item
        for assessment in assessments
        for item in (
            *assessment["criterion_assessments"],
            *assessment["late_obligation_candidates"],
        )
    ]
    assert {item["implementation_status"] for item in records} == {
        "covered",
        "partial",
        "missing",
        "uncertain",
        "not_applicable",
    }
    assert {assessment["verdict"] for assessment in assessments} == {
        "approve",
        "request_changes",
    }
    assert {
        item["action_kind"] for assessment in assessments for item in assessment["repair_requests"]
    } == {
        "fix",
        "investigate",
    }
    assert any(item["implementation_evidence"] for item in records)
    assert any(item["test_evidence"] for item in records)
    assert any(assessment["additional_sources"] for assessment in assessments)
    assert any(assessment["unexpected_regressions"] for assessment in assessments)
    assert any(assessment["late_obligation_candidates"] for assessment in assessments)


def test_source_grounded_v3_few_shots_cover_obligation_contract_and_repair_boundaries():
    root = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v3"
    frozen = load_frozen_few_shots(root)
    provenance = json.loads((root / "provenance.json").read_text())

    assert frozen.contract_version == "3.0"
    assert frozen.case_ids == (
        "synthetic-retry-delay",
        "synthetic-format-codec",
        "synthetic-policy-boundary",
    )
    assert set(frozen.case_ids).isdisjoint(DEVELOPMENT_TASK_IDS)
    assert set(frozen.case_ids).isdisjoint(EVALUATION_TASK_IDS)
    assert provenance["synthetic_only"] is True
    assert provenance["uses_benchmark_artifacts"] is False

    validator = ContractValidator(DEFAULT_SCHEMA_DIR, contract_version="3.0")
    ledgers = {}
    assessments = []
    audits = {}
    for case_id in frozen.case_ids:
        case = root / case_id
        stage_a_input = json.loads((case / "stage_a_input.json").read_text())
        ledger = json.loads((case / "stage_a_output.json").read_text())
        assessment = json.loads((case / "stage_b_output.json").read_text())
        audit = json.loads((case / "repair_audit_output.json").read_text())
        validator.validate_schema("build_input", stage_a_input)
        validator.validate_ledger(ledger, task_id=case_id)
        validator.validate_task_source_anchors(
            ledger["sources"],
            title=stage_a_input["task"]["title"],
            description=stage_a_input["task"]["description"],
        )
        validator.validate_assessment(
            assessment,
            task_id=case_id,
            patch_sha256=assessment["patch_sha256"],
            ledger_sha256=assessment["ledger_sha256"],
            ledger=ledger,
        )
        validator.validate_schema("repair_audit", audit)
        ledgers[case_id] = ledger
        assessments.append(assessment)
        audits[case_id] = audit

    format_assessment = next(
        item for item in assessments if item["task_id"] == "synthetic-format-codec"
    )
    effective = validator.build_effective_review_set(
        task_id="synthetic-format-codec",
        ledger_sha256=format_assessment["ledger_sha256"],
        assessment_sha256=audits["synthetic-format-codec"]["assessment_sha256"],
        ledger=ledgers["synthetic-format-codec"],
        assessment=format_assessment,
    )
    validator.validate_repair_audit(
        audits["synthetic-format-codec"],
        task_id="synthetic-format-codec",
        p0_sha256=format_assessment["patch_sha256"],
        assessment_sha256=audits["synthetic-format-codec"]["assessment_sha256"],
        effective_review_set_sha256=audits["synthetic-format-codec"]["effective_review_set_sha256"],
        ledger=ledgers["synthetic-format-codec"],
        assessment=format_assessment,
        effective_review_set=effective,
        actual_changed_paths={"src/codec.py"},
    )

    records = [
        record for assessment in assessments for record in assessment["obligation_assessments"]
    ]
    assert {assessment["verdict"] for assessment in assessments} == {
        "approve",
        "request_changes",
        "abstain",
    }
    assert {record["source_status"] for record in records} == {
        "supported",
        "unsupported",
        "unclear",
    }
    assert {record["implementation_status"] for record in records} == {
        "covered",
        "missing",
        "partial",
        "uncertain",
        "not_applicable",
    }
    assert any(assessment["late_obligations"] for assessment in assessments)
    assert any(
        request["action"] == "fix"
        for assessment in assessments
        for request in assessment["repair_requests"]
    )
    policy_summary = json.loads(
        (root / "synthetic-policy-boundary/repair_input_summary.json").read_text()
    )
    assert policy_summary["counterfactual_request"]["action"] == "investigate"
    assert audits["synthetic-policy-boundary"]["changes"][0]["decision"] == "no_change"


def test_source_grounded_v4_few_shots_pin_prompt_and_preserve_v3_artifacts():
    root = Path(__file__).resolve().parents[1]
    v3_root = root / "few_shots/source_grounded_v3"
    v4_root = root / "few_shots/source_grounded_v4"
    v3 = load_frozen_few_shots(v3_root)
    v4 = load_frozen_few_shots(v4_root)

    assert v3.prompt_version is None
    assert v4.contract_version == "3.0"
    assert v4.prompt_version == "4.0"
    assert v4.case_ids == v3.case_ids


def test_few_shot_provenance_rejects_invalid_prompt_version(tmp_path):
    source = Path(__file__).resolve().parents[1] / "few_shots/source_grounded_v4"
    destination = tmp_path / "few-shots"
    shutil.copytree(source, destination)
    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["prompt_version"] = "5.0"
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="Invalid few-shot prompt version"):
        load_frozen_few_shots(destination)


def test_random60_scope_freezes_all_source_tasks_and_excludes_development_metrics(
    tmp_path,
):
    task_ids = (DEVELOPMENT_TASK_IDS[0], "task-a", DEVELOPMENT_TASK_IDS[1])
    source_manifest = tmp_path / "random60.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "deepswe",
                "benchmark_revision": "1.1",
                "task_count": len(task_ids),
                "task_ids": list(task_ids),
            }
        )
    )
    source_job = tmp_path / "source-job"
    source_job.mkdir()
    (tmp_path / "source-job.run.json").write_text(
        json.dumps(
            {
                "benchmark_revision": "1.1",
                "agent": "pi",
                "manifest": str(source_manifest),
            }
        )
    )
    checkout = tmp_path / "checkout"
    for task_id in task_ids:
        trial = source_job / f"{task_id}__trial"
        (trial / "artifacts").mkdir(parents=True)
        (trial / "config.json").write_text(
            json.dumps(
                {
                    "job_id": "job-1",
                    "task": {"path": f"/tasks/{task_id}"},
                }
            )
        )
        (trial / "artifacts" / "model.patch").write_text(f"patch for {task_id}")
        environment = checkout / "tasks" / task_id / "environment"
        environment.mkdir(parents=True)
        (environment / "Dockerfile").write_text("ARG BASE_SHA=" + "a" * 40 + "\n")

    output = tmp_path / "review-random60.json"
    build_reviewer_experiment_manifest(
        source_job_dir=source_job,
        checkout=checkout,
        output=output,
        scope="random60",
    )
    loaded = load_reviewer_experiment_manifest(output)

    assert loaded.task_ids == task_ids
    assert loaded.metric_task_ids == ("task-a",)
    assert loaded.excluded_task_ids == DEVELOPMENT_TASK_IDS
    assert set(loaded.patches) == set(task_ids)
    assert loaded.raw["selection_method"].startswith("source-manifest-all-tasks")
