import hashlib
import json

from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest
from external_agents_eval.benchmarks.deepswe.report import (
    collect_reviewer_experiment_summary,
    collect_summary,
)
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    ReviewerExperimentManifest,
)
from external_agents_eval.workflows.solve_review_repair.models import ExistingPatch


def _manifest(tmp_path):
    return DeepSWEManifest(
        path=tmp_path / "manifest.json",
        task_ids=("passed", "failed", "pending"),
        sample_seed=1,
        sample_size=3,
        population_count=113,
        population_fingerprint="a" * 64,
        raw={},
    )


def _write_result(job_dir, trial, task, reward, *, filename="result.json", namespaced=True):
    path = job_dir / trial
    path.mkdir(parents=True)
    task_name = f"datacurve/{task}" if namespaced else task
    (path / filename).write_text(
        json.dumps(
            {
                "task_name": task_name,
                "finished_at": "2026-09-01T00:00:00Z",
                "verifier_result": {"rewards": {"reward": reward}},
            }
        )
    )


def test_collect_deepswe_summary(tmp_path):
    job_dir = tmp_path / "job"
    _write_result(job_dir, "passed__one", "passed", 1)
    _write_result(
        job_dir,
        "failed__one",
        "failed",
        0,
        filename="results.json",
        namespaced=False,
    )
    summary = collect_summary(job_dir, _manifest(tmp_path))
    assert summary["counts"] == {"failed": 1, "passed": 1, "pending": 1}
    assert summary["evaluated_count"] == 2
    assert summary["pass_rate"] is None
    assert summary["evaluated_pass_rate"] == 0.5


def test_collect_reviewer_summary_reports_recovery_not_overall_uplift(tmp_path):
    source_job = tmp_path / "source"
    target_job = tmp_path / "target"
    patches = {}
    for task_id, baseline_partial in (("recovered", 0.75), ("still-failing", 0.5)):
        source_trial = source_job / f"{task_id}__source"
        (source_trial / "verifier").mkdir(parents=True)
        (source_trial / "verifier" / "reward.json").write_text(
            json.dumps(
                {
                    "reward": 0,
                    "partial": baseline_partial,
                    "f2p_passed": 1,
                    "p2p_passed": 3,
                }
            )
        )
        patch_path = tmp_path / f"{task_id}.patch"
        patch_path.write_text("P0")
        patches[task_id] = ExistingPatch(
            task_id=task_id,
            base_commit="a" * 40,
            path=patch_path,
            sha256="a" * 64,
            tracked_diff_sha256="a" * 64,
            source_job_id="job-1",
            source_trial_id=source_trial.name,
        )
    experiment = ReviewerExperimentManifest(
        path=tmp_path / "manifest.json",
        source_job_id="job-1",
        source_job_dir=source_job,
        source_manifest_path=tmp_path / "source-manifest.json",
        source_manifest_sha256="b" * 64,
        task_ids=("recovered", "still-failing"),
        patches=patches,
        raw={},
    )

    for task_id, reward, partial, verdict in (
        ("recovered", 1, 1.0, "request_changes"),
        ("still-failing", 0, 0.75, "approve"),
    ):
        trial = target_job / f"{task_id}__new"
        review = trial / "review_repair"
        (review / "reviewer").mkdir(parents=True)
        (review / "inputs").mkdir()
        (review / "outputs").mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_id,
                    "finished_at": "2026-09-04T00:00:00Z",
                    "verifier_result": {
                        "rewards": {
                            "reward": reward,
                            "partial": partial,
                            "f2p_passed": 2,
                            "p2p_passed": 3,
                        }
                    },
                }
            )
        )
        (review / "manifest.json").write_text(
            json.dumps(
                {
                    "final_state": "COMPLETED",
                    "reviewer_verdict": verdict,
                    "repair_executed": verdict != "approve",
                    "p0_sha256": "a" * 64,
                    "p1_sha256": hashlib.sha256(b"P1-longer").hexdigest(),
                }
            )
        )
        (review / "reviewer" / "stage_a_validation.json").write_text(
            json.dumps(
                {
                    "valid": True,
                    "correction_used": False,
                    "normalization_used": True,
                }
            )
        )
        (review / "reviewer" / "stage_b_validation.json").write_text(
            json.dumps(
                {
                    "valid": True,
                    "correction_used": True,
                    "normalization_used": False,
                }
            )
        )
        (review / "reviewer" / "review_obligations.json").write_text(
            json.dumps({"obligations": [{"category": "public_api"}]})
        )
        (review / "reviewer" / "patch_assessment.json").write_text(
            json.dumps(
                {
                    "obligation_assessments": [{"status": "covered"}],
                    "late_obligation_candidates": [],
                    "unexpected_regressions": [],
                    "repair_requests": [],
                }
            )
        )
        (review / "inputs" / "p0.patch").write_text("P0")
        (review / "outputs" / "p1.patch").write_text("P1-longer")

    summary = collect_reviewer_experiment_summary(target_job, experiment)
    assert summary["metric_name"] == "recovery_rate_not_overall_pass_rate_uplift"
    assert summary["recovery_rate"] == 0.5
    assert summary["transitions"] == {
        "fail_to_fail_score_improved": 1,
        "fail_to_pass": 1,
    }
    assert summary["false_approve_rate"] == 1.0
    assert summary["schema_validity"]["stage_a_first_attempt"] == 0.0
    assert summary["schema_validity"]["stage_a_after_normalization"] == 1.0
    assert summary["schema_validity"]["stage_b_first_attempt"] == 0.0
    assert summary["schema_validity"]["stage_b_after_normalization"] == 0.0
    assert summary["schema_validity"]["stage_b_after_correction"] == 1.0
    assert summary["f2p_recovered_count"] == 2
    assert summary["p2p_regressed_count"] == 0


def test_collect_reviewer_summary_reports_paired_pass_rate_and_exclusions(tmp_path):
    source_job = tmp_path / "source"
    target_job = tmp_path / "target"
    task_ids = ("baseline-pass", "baseline-fail", "development-case")
    patches = {}
    for task_id, reward in zip(task_ids, (1, 0, 1), strict=True):
        source_trial = source_job / f"{task_id}__source"
        (source_trial / "verifier").mkdir(parents=True)
        (source_trial / "verifier" / "reward.json").write_text(
            json.dumps(
                {
                    "reward": reward,
                    "partial": float(reward),
                    "f2p_passed": reward,
                    "p2p_passed": 3,
                }
            )
        )
        patch_path = tmp_path / f"{task_id}.patch"
        patch_path.write_text("P0")
        patches[task_id] = ExistingPatch(
            task_id=task_id,
            base_commit="a" * 40,
            path=patch_path,
            sha256="a" * 64,
            tracked_diff_sha256="a" * 64,
            source_job_id="job-1",
            source_trial_id=source_trial.name,
        )
    experiment = ReviewerExperimentManifest(
        path=tmp_path / "manifest.json",
        source_job_id="job-1",
        source_job_dir=source_job,
        source_manifest_path=tmp_path / "source-manifest.json",
        source_manifest_sha256="b" * 64,
        task_ids=task_ids,
        patches=patches,
        raw={},
        metric_task_ids=("baseline-pass", "baseline-fail"),
        excluded_task_ids=("development-case",),
    )

    for task_id, reward in zip(task_ids, (0, 1, 1), strict=True):
        trial = target_job / f"{task_id}__new"
        review = trial / "review_repair"
        (review / "inputs").mkdir(parents=True)
        (review / "outputs").mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_id,
                    "finished_at": "2026-09-05T00:00:00Z",
                    "verifier_result": {
                        "rewards": {
                            "reward": reward,
                            "partial": float(reward),
                            "f2p_passed": reward,
                            "p2p_passed": 3,
                        }
                    },
                }
            )
        )
        (review / "manifest.json").write_text(
            json.dumps(
                {
                    "final_state": "COMPLETED",
                    "reviewer_verdict": "request_changes",
                    "repair_executed": True,
                    "p1_sha256": hashlib.sha256(b"P1").hexdigest(),
                }
            )
        )
        (review / "inputs" / "p0.patch").write_text("P0")
        (review / "outputs" / "p1.patch").write_text("P1")

    summary = collect_reviewer_experiment_summary(target_job, experiment)

    assert summary["metric_name"] == "paired_p0_to_p1_pass_rate"
    assert summary["task_count"] == 3
    assert summary["metric_task_count"] == 2
    assert summary["excluded_task_count"] == 1
    assert summary["p0_pass_rate"] == 0.5
    assert summary["p1_pass_rate"] == 0.5
    assert summary["net_pass_change"] == 0
    assert summary["recovery_rate"] == 1.0
    assert summary["retention_rate"] == 0.0
    assert summary["transitions"] == {"fail_to_pass": 1, "pass_to_fail": 1}
    assert summary["excluded_transitions"] == {"pass_to_pass": 1}


def test_reviewer_summary_rejects_verifier_result_without_workflow_p1_manifest(tmp_path):
    source_trial = tmp_path / "source" / "task-a__source"
    (source_trial / "verifier").mkdir(parents=True)
    (source_trial / "verifier" / "reward.json").write_text(
        json.dumps({"reward": 0, "partial": 0.5})
    )
    p0_path = tmp_path / "p0.patch"
    p0_path.write_bytes(b"P0")
    experiment = ReviewerExperimentManifest(
        path=tmp_path / "manifest.json",
        source_job_id="job-1",
        source_job_dir=tmp_path / "source",
        source_manifest_path=tmp_path / "source-manifest.json",
        source_manifest_sha256="b" * 64,
        task_ids=("task-a",),
        patches={
            "task-a": ExistingPatch(
                task_id="task-a",
                base_commit="a" * 40,
                path=p0_path,
                sha256=hashlib.sha256(b"P0").hexdigest(),
                tracked_diff_sha256=hashlib.sha256(b"P0").hexdigest(),
                source_job_id="job-1",
                source_trial_id=source_trial.name,
            )
        },
        raw={},
    )
    trial = tmp_path / "target" / "task-a__new"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "finished_at": "2026-09-05T00:00:00Z",
                "verifier_result": {"rewards": {"reward": 0, "partial": 0.0}},
            }
        )
    )

    summary = collect_reviewer_experiment_summary(tmp_path / "target", experiment)

    assert summary["evaluated_count"] == 0
    assert summary["transitions"] == {"infrastructure_error": 1}
    assert summary["infrastructure_statuses"] == {"infra_failed_terminal": 1}
    assert summary["failure_classes"] == {"workflow_pairing": 1}
    assert summary["details"]["task-a"]["valid_p1_pair"] is False
    assert summary["details"]["task-a"]["infrastructure_status"] == (
        "infra_failed_terminal"
    )
    assert "missing_or_invalid_workflow_manifest" in summary["details"]["task-a"]["pairing_errors"]


def test_reviewer_summary_counts_internal_repair_timeout_from_published_patch(tmp_path):
    source_trial = tmp_path / "source" / "task-a__source"
    (source_trial / "verifier").mkdir(parents=True)
    (source_trial / "verifier" / "reward.json").write_text(
        json.dumps({"reward": 0, "partial": 0.5})
    )
    p0_path = tmp_path / "p0.patch"
    p0_path.write_bytes(b"P0")
    experiment = ReviewerExperimentManifest(
        path=tmp_path / "manifest.json",
        source_job_id="job-1",
        source_job_dir=tmp_path / "source",
        source_manifest_path=tmp_path / "source-manifest.json",
        source_manifest_sha256="b" * 64,
        task_ids=("task-a",),
        patches={
            "task-a": ExistingPatch(
                task_id="task-a",
                base_commit="a" * 40,
                path=p0_path,
                sha256=hashlib.sha256(b"P0").hexdigest(),
                tracked_diff_sha256=hashlib.sha256(b"P0").hexdigest(),
                source_job_id="job-1",
                source_trial_id=source_trial.name,
            )
        },
        raw={},
    )
    trial = tmp_path / "target" / "task-a__new"
    review = trial / "review_repair"
    (review / "outputs").mkdir(parents=True)
    (review / "inputs").mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "finished_at": "2026-09-05T00:00:00Z",
                "verifier_result": {"rewards": {"reward": 0, "partial": 0.25}},
            }
        )
    )
    (review / "inputs" / "p0.patch").write_bytes(b"P0")
    (review / "outputs" / "p1.patch").write_bytes(b"partial P1")
    (review / "manifest.json").write_text(
        json.dumps(
            {
                "final_state": "REPAIR_FAILED",
                "p1_sha256": hashlib.sha256(b"partial P1").hexdigest(),
                "repair_executed": True,
            }
        )
    )

    summary = collect_reviewer_experiment_summary(tmp_path / "target", experiment)

    assert summary["evaluated_count"] == 1
    assert summary["transitions"] == {"fail_to_fail_score_declined": 1}
    assert summary["details"]["task-a"]["valid_p1_pair"] is True


def test_reviewer_summary_reports_v2_abstain_source_and_audit_metrics(tmp_path):
    source_trial = tmp_path / "source" / "task-v2__source"
    (source_trial / "verifier").mkdir(parents=True)
    (source_trial / "verifier/reward.json").write_text(
        json.dumps({"reward": 0, "partial": 0.5, "f2p_passed": 1, "p2p_passed": 3})
    )
    patch_path = tmp_path / "p0.patch"
    patch_path.write_bytes(b"P0")
    experiment = ReviewerExperimentManifest(
        path=tmp_path / "manifest.json",
        source_job_id="job-v2",
        source_job_dir=tmp_path / "source",
        source_manifest_path=tmp_path / "source-manifest.json",
        source_manifest_sha256="b" * 64,
        task_ids=("task-v2",),
        patches={
            "task-v2": ExistingPatch(
                task_id="task-v2",
                base_commit="a" * 40,
                path=patch_path,
                sha256=hashlib.sha256(b"P0").hexdigest(),
                tracked_diff_sha256=hashlib.sha256(b"P0").hexdigest(),
                source_job_id="job-v2",
                source_trial_id=source_trial.name,
            )
        },
        raw={},
    )
    trial = tmp_path / "target/task-v2__run"
    workflow = trial / "review_repair"
    (workflow / "inputs").mkdir(parents=True)
    (workflow / "outputs").mkdir()
    (workflow / "reviewer").mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-v2",
                "finished_at": "2026-09-07T00:00:00Z",
                "verifier_result": {
                    "rewards": {
                        "reward": 0,
                        "partial": 0.5,
                        "f2p_passed": 1,
                        "p2p_passed": 3,
                    }
                },
            }
        )
    )
    p1 = b"P0"
    (workflow / "inputs/p0.patch").write_bytes(p1)
    (workflow / "outputs/p1.patch").write_bytes(p1)
    (workflow / "manifest.json").write_text(
        json.dumps(
            {
                "final_state": "COMPLETED",
                "reviewer_verdict": "abstain",
                "repair_executed": False,
                "repair_audit_status": "not_applicable",
                "repair_sequence_observation_status": "not_applicable",
                "contract_version": "2.0",
                "experiment_arm": "E2",
                "abstain_reason": "A key public contract conflicts.",
                "p0_sha256": hashlib.sha256(p1).hexdigest(),
                "p1_sha256": hashlib.sha256(p1).hexdigest(),
            }
        )
    )
    for stage in ("stage_a", "stage_b"):
        (workflow / f"reviewer/{stage}_validation.json").write_text(
            json.dumps({"valid": True, "correction_used": False, "normalization_used": False})
        )
    (workflow / "reviewer/source_anchor_validation.json").write_text(json.dumps({"valid": True}))
    (workflow / "reviewer/review_obligations.json").write_text(
        json.dumps({"obligations": [{"category": "public_api"}]})
    )
    (workflow / "reviewer/patch_assessment.json").write_text(
        json.dumps(
            {
                "criterion_assessments": [
                    {
                        "implementation_status": "uncertain",
                        "source_check": {"status": "conflicting"},
                    }
                ],
                "ledger_challenges": [{"kind": "source_conflict", "disposition": "unresolved"}],
                "late_obligation_candidates": [],
                "unexpected_regressions": [],
                "repair_requests": [],
            }
        )
    )
    (workflow / "reviewer/effective_review_set.json").write_text(
        json.dumps(
            {
                "effective_criteria": [],
                "unresolved_criteria": ["O01.C01"],
                "repair_request_ids": [],
            }
        )
    )
    (workflow / "outputs/resource_usage.json").write_text(
        json.dumps(
            {
                "roles": [
                    {
                        "role": "patch_reviewer",
                        "timed_out": False,
                        "resource_usage": {"wall_time_sec": 12.5},
                    }
                ]
            }
        )
    )

    summary = collect_reviewer_experiment_summary(tmp_path / "target", experiment)
    assert summary["approval_coverage"] == 0.0
    assert summary["abstain_rate"] == 1.0
    assert summary["source_anchor_validity"] == 1.0
    assert summary["criterion_source_checks"] == {"conflicting": 1}
    assert summary["ledger_challenge_dispositions"] == {"unresolved": 1}
    assert summary["unresolved_criterion_count"] == 1
    assert summary["repair_audit_statuses"] == {"not_applicable": 1}
    assert summary["repair_sequence_observation_statuses"] == {"not_applicable": 1}
    assert summary["role_resources"]["patch_reviewer"] == {
        "runs": 1,
        "timed_out": 0,
        "wall_time_sec": 12.5,
    }
    assert summary["details"]["task-v2"]["abstain_reason"] == ("A key public contract conflicts.")


def test_reviewer_summary_reports_v3_obligation_source_and_omission_metrics(tmp_path):
    source_trial = tmp_path / "source/task-v3__source"
    (source_trial / "verifier").mkdir(parents=True)
    (source_trial / "verifier/reward.json").write_text(json.dumps({"reward": 0, "partial": 0.5}))
    p0_path = tmp_path / "p0.patch"
    p0_path.write_bytes(b"P0")
    experiment = ReviewerExperimentManifest(
        path=tmp_path / "manifest.json",
        source_job_id="job-v3",
        source_job_dir=tmp_path / "source",
        source_manifest_path=tmp_path / "source-manifest.json",
        source_manifest_sha256="b" * 64,
        task_ids=("task-v3",),
        patches={
            "task-v3": ExistingPatch(
                task_id="task-v3",
                base_commit="a" * 40,
                path=p0_path,
                sha256=hashlib.sha256(b"P0").hexdigest(),
                tracked_diff_sha256=hashlib.sha256(b"P0").hexdigest(),
                source_job_id="job-v3",
                source_trial_id=source_trial.name,
            )
        },
        raw={},
    )
    trial = tmp_path / "target/task-v3__run"
    workflow = trial / "review_repair"
    (workflow / "inputs").mkdir(parents=True)
    (workflow / "outputs").mkdir()
    (workflow / "reviewer").mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-v3",
                "finished_at": "2026-09-08T00:00:00Z",
                "verifier_result": {"rewards": {"reward": 0, "partial": 0.5}},
            }
        )
    )
    (workflow / "inputs/p0.patch").write_bytes(b"P0")
    (workflow / "outputs/p1.patch").write_bytes(b"P0")
    digest = hashlib.sha256(b"P0").hexdigest()
    (workflow / "manifest.json").write_text(
        json.dumps(
            {
                "final_state": "COMPLETED",
                "reviewer_verdict": "abstain",
                "repair_executed": False,
                "contract_version": "3.0",
                "prompt_version": "4.0",
                "experiment_arm": "E2",
                "abstain_reason": "One must obligation is unclear.",
                "p0_sha256": digest,
                "p1_sha256": digest,
            }
        )
    )
    (workflow / "reviewer/patch_assessment.json").write_text(
        json.dumps(
            {
                "obligation_assessments": [
                    {
                        "obligation_ref": "O01",
                        "source_status": "supported",
                        "implementation_status": "covered",
                    },
                    {
                        "obligation_ref": "O02",
                        "source_status": "unsupported",
                        "implementation_status": "not_applicable",
                    },
                    {
                        "obligation_ref": "O03",
                        "source_status": "unclear",
                        "implementation_status": "not_applicable",
                    },
                    {
                        "obligation_ref": "L01",
                        "source_status": "supported",
                        "implementation_status": "missing",
                    },
                ],
                "late_obligations": [{"id": "L01"}],
                "unexpected_regressions": [],
                "repair_requests": [],
            }
        )
    )
    (workflow / "reviewer/effective_review_set.json").write_text(
        json.dumps(
            {
                "effective_obligations": ["O01", "L01"],
                "nonbinding_obligations": ["O02"],
                "unresolved_obligations": ["O03"],
                "repair_request_ids": [],
            }
        )
    )

    summary = collect_reviewer_experiment_summary(tmp_path / "target", experiment)

    assert summary["assessment_statuses"] == {"covered": 1, "missing": 1, "not_applicable": 2}
    assert summary["obligation_source_statuses"] == {
        "supported": 2,
        "unclear": 1,
        "unsupported": 1,
    }
    assert summary["effective_obligation_count"] == 2
    assert summary["nonbinding_obligation_count"] == 1
    assert summary["unresolved_obligation_count"] == 1
    assert summary["late_obligation_count"] == 1
    assert summary["stage_a_omission_count"] == 1
    assert summary["details"]["task-v3"]["effective_obligation_count"] == 2
    assert summary["details"]["task-v3"]["late_obligation_count"] == 1
    assert summary["details"]["task-v3"]["contract_version"] == "3.0"
    assert summary["details"]["task-v3"]["prompt_version"] == "4.0"
    assert summary["workflow_versions"] == {"contract=3.0;prompt=4.0": 1}


def test_reviewer_summary_accepts_v5_publication_layout(tmp_path):
    source_trial = tmp_path / "source/task-v5__source"
    (source_trial / "verifier").mkdir(parents=True)
    (source_trial / "verifier/reward.json").write_text(
        json.dumps({"reward": 0, "partial": 0.5})
    )
    p0 = b"P0"
    p1 = b"P1"
    p0_path = tmp_path / "p0.patch"
    p0_path.write_bytes(p0)
    experiment = ReviewerExperimentManifest(
        path=tmp_path / "manifest.json",
        source_job_id="job-v5",
        source_job_dir=tmp_path / "source",
        source_manifest_path=tmp_path / "source-manifest.json",
        source_manifest_sha256="b" * 64,
        task_ids=("task-v5",),
        patches={
            "task-v5": ExistingPatch(
                task_id="task-v5",
                base_commit="a" * 40,
                path=p0_path,
                sha256=hashlib.sha256(p0).hexdigest(),
                tracked_diff_sha256=hashlib.sha256(p0).hexdigest(),
                source_job_id="job-v5",
                source_trial_id=source_trial.name,
            )
        },
        raw={},
    )
    trial = tmp_path / "target/task-v5__run"
    workflow = trial / "v5_review_repair"
    (workflow / "outputs").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-v5",
                "finished_at": "2026-09-11T00:00:00Z",
                "verifier_result": {
                    "rewards": {"reward": 1, "partial": 1.0}
                },
            }
        )
    )
    (workflow / "outputs/v5-p0.patch").write_bytes(p0)
    (workflow / "outputs/v5-p1.patch").write_bytes(p1)
    (workflow / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v5-run-1",
                "final_state": "COMPLETED",
                "workflow_state": "completed",
                "route": "repair",
                "repair_executed": True,
                "contract_version": "5.0",
                "prompt_version": "v5-1",
                "p0_sha256": hashlib.sha256(p0).hexdigest(),
                "p1_sha256": hashlib.sha256(p1).hexdigest(),
            }
        )
    )

    summary = collect_reviewer_experiment_summary(
        tmp_path / "target", experiment
    )

    assert summary["evaluated_count"] == 1
    assert summary["transitions"] == {"fail_to_pass": 1}
    assert summary["workflow_routes"] == {"repair": 1}
    assert summary["workflow_states"] == {"completed": 1}
    assert summary["workflow_versions"] == {"contract=5.0;prompt=v5-1": 1}
    assert summary["details"]["task-v5"]["valid_p1_pair"] is True
    assert summary["details"]["task-v5"]["workflow_route"] == "repair"
