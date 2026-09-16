from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest
from external_agents_eval.benchmarks.deepswe.pier_compat import (
    _is_transient_environment_start_error,
)
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    ReviewerExperimentManifest,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    TERMINAL_FAILURE_STATES,
    WorkflowState,
)

PUBLISHED_WORKFLOW_STATES = frozenset(
    {WorkflowState.COMPLETED.value, *(state.value for state in TERMINAL_FAILURE_STATES)}
)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _task_id(result: dict[str, Any], selected: set[str]) -> str | None:
    candidates: list[str] = []
    task_name = result.get("task_name")
    if isinstance(task_name, str):
        candidates.extend((task_name, task_name.rsplit("/", 1)[-1]))

    task_id = result.get("task_id")
    config = result.get("config")
    for value in (
        task_id.get("path") if isinstance(task_id, dict) else None,
        config.get("task", {}).get("path") if isinstance(config, dict) else None,
    ):
        if isinstance(value, str):
            candidates.append(Path(value).name)

    return next((candidate for candidate in candidates if candidate in selected), None)


def _published_p1_errors(
    *,
    workflow_manifest: dict[str, Any] | None,
    p1_path: Path,
    p1_reward: Any,
) -> list[str]:
    errors: list[str] = []
    if workflow_manifest is None:
        errors.append("missing_or_invalid_workflow_manifest")
        manifest: dict[str, Any] = {}
    else:
        manifest = workflow_manifest
    if manifest.get("final_state") not in PUBLISHED_WORKFLOW_STATES:
        errors.append("workflow_did_not_reach_published_terminal_state")
    if not p1_path.is_file():
        errors.append("missing_p1_patch")
    else:
        actual_sha = hashlib.sha256(p1_path.read_bytes()).hexdigest()
        if manifest.get("p1_sha256") != actual_sha:
            errors.append("p1_sha256_mismatch")
    if not isinstance(p1_reward, (int, float)):
        errors.append("missing_verifier_reward")
    return errors


def _infrastructure_failure_metadata(
    *,
    trial_dir: Path,
    result: dict[str, Any],
    transition: str,
) -> dict[str, Any]:
    if transition != "infrastructure_error":
        return {
            "infrastructure_status": None,
            "failure_class": None,
            "infra_start_attempts": None,
        }
    start = _load_json(trial_dir / "environment_start.json") or {}
    start_history = _load_json(trial_dir / "environment_start_attempts.json") or {}
    verifier = _load_json(trial_dir / "verifier_observability.json") or {}
    verifier_class = verifier.get("original_failure_class") or verifier.get("last_failure_class")
    if isinstance(verifier_class, str) and verifier_class not in {"none", "test_failure"}:
        return {
            "infrastructure_status": (
                "infra_failed_retryable"
                if verifier_class in {"verifier_timeout", "verifier_infrastructure_flake"}
                else "infra_failed_terminal"
            ),
            "failure_class": verifier_class,
            "infra_start_attempts": start.get("attempts"),
            "infra_start_attempt_history": start_history.get("attempts"),
            "verifier_retry_count": verifier.get("retry_count"),
            "verifier_attempt_count": verifier.get("attempt_count"),
        }
    recorded_status = start.get("status")
    if recorded_status in {"infra_failed_retryable", "infra_failed_terminal"}:
        return {
            "infrastructure_status": recorded_status,
            "failure_class": start.get("failure_class") or "environment_start",
            "infra_start_attempts": start.get("attempts"),
            "infra_start_attempt_history": start_history.get("attempts"),
            "verifier_retry_count": verifier.get("retry_count"),
            "verifier_attempt_count": verifier.get("attempt_count"),
        }
    exception = result.get("exception_info")
    exception_text = ""
    if isinstance(exception, dict):
        exception_text = "\n".join(
            str(exception.get(key) or "")
            for key in ("exception_type", "exception_message", "exception_traceback")
        )
    elif exception:
        exception_text = str(exception)
    if _is_transient_environment_start_error(exception_text):
        return {
            "infrastructure_status": "infra_failed_retryable",
            "failure_class": "environment_start",
            "infra_start_attempts": None,
            "infra_start_attempt_history": start_history.get("attempts"),
            "verifier_retry_count": verifier.get("retry_count"),
            "verifier_attempt_count": verifier.get("attempt_count"),
        }
    failure_class = (
        "reviewer_input_preflight"
        if "p0 manifest base commit" in exception_text.lower()
        else "workflow_pairing"
    )
    return {
        "infrastructure_status": "infra_failed_terminal",
        "failure_class": failure_class,
        "infra_start_attempts": None,
        "infra_start_attempt_history": start_history.get("attempts"),
        "verifier_retry_count": verifier.get("retry_count"),
        "verifier_attempt_count": verifier.get("attempt_count"),
    }


def _verifier_observability(trial_dir: Path) -> dict[str, Any]:
    payload = _load_json(trial_dir / "verifier_observability.json") or {}
    return {
        "verifier_failure_class": payload.get("original_failure_class"),
        "verifier_last_failure_class": payload.get("last_failure_class"),
        "verifier_attempt_count": payload.get("attempt_count"),
        "verifier_retry_count": payload.get("retry_count"),
    }


def collect_summary(job_dir: str | Path, manifest: DeepSWEManifest) -> dict[str, Any]:
    root = Path(job_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Pier job directory not found: {root}")
    selected = set(manifest.task_ids)
    trials_by_task: dict[str, list[dict[str, Any]]] = {}
    extra_tasks: set[str] = set()
    result_paths = set(root.glob("*/result.json"))
    result_paths.update(root.glob("*/results.json"))  # Pier < 0.3 compatibility
    for result_path in sorted(result_paths):
        result = _load_json(result_path)
        if result is None:
            continue
        task_name = _task_id(result, selected)
        if task_name is None:
            raw_task_name = result.get("task_name")
            if isinstance(raw_task_name, str):
                extra_tasks.add(raw_task_name)
            continue
        result["_trial_dir"] = str(result_path.parent)
        trials_by_task.setdefault(task_name, []).append(result)

    counts: Counter[str] = Counter()
    details: dict[str, Any] = {}
    rewards: list[float] = []
    for task_id in manifest.task_ids:
        trials = trials_by_task.get(task_id, [])
        if not trials:
            counts["pending"] += 1
            details[task_id] = {"status": "pending"}
            continue
        latest = max(
            trials,
            key=lambda value: str(value.get("finished_at") or value.get("started_at") or ""),
        )
        verifier = latest.get("verifier_result") or {}
        reward_map = verifier.get("rewards") if isinstance(verifier, dict) else None
        reward = reward_map.get("reward") if isinstance(reward_map, dict) else None
        exception = latest.get("exception_info")
        if isinstance(reward, (int, float)):
            numeric_reward = float(reward)
            rewards.append(numeric_reward)
            status = "passed" if numeric_reward == 1.0 else "failed"
        elif exception:
            status = "error"
        else:
            status = "incomplete"
        counts[status] += 1
        details[task_id] = {
            "status": status,
            "reward": reward,
            "rewards": reward_map,
            "trial_dir": latest["_trial_dir"],
            "attempt_count": len(trials),
            "exception": exception,
            **_verifier_observability(Path(latest["_trial_dir"])),
        }
    evaluated_pass_rate = (sum(rewards) / len(rewards)) if rewards else None
    return {
        "benchmark": "deepswe",
        "benchmark_revision": "1.1",
        "job_dir": str(root),
        "task_count": len(manifest.task_ids),
        "evaluated_count": len(rewards),
        "pass_rate": (evaluated_pass_rate if len(rewards) == len(manifest.task_ids) else None),
        "evaluated_pass_rate": evaluated_pass_rate,
        "counts": dict(sorted(counts.items())),
        "extra_tasks": sorted(extra_tasks),
        "details": details,
    }


def collect_reviewer_experiment_summary(
    job_dir: str | Path,
    manifest: ReviewerExperimentManifest,
) -> dict[str, Any]:
    """Summarize paired P0/P1 outcomes for a reviewer-repair experiment."""
    root = Path(job_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Pier job directory not found: {root}")
    selected = set(manifest.task_ids)
    trials_by_task: dict[str, list[dict[str, Any]]] = {}
    result_paths = set(root.glob("*/result.json"))
    result_paths.update(root.glob("*/results.json"))
    for result_path in sorted(result_paths):
        result = _load_json(result_path)
        if result is None:
            continue
        task_id = _task_id(result, selected)
        if task_id is None:
            continue
        result["_trial_dir"] = str(result_path.parent)
        trials_by_task.setdefault(task_id, []).append(result)

    metric_task_ids = set(manifest.metric_task_ids or manifest.task_ids)
    excluded_task_ids = set(manifest.excluded_task_ids)
    transitions: Counter[str] = Counter()
    excluded_transitions: Counter[str] = Counter()
    workflow_states: Counter[str] = Counter()
    workflow_routes: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    workflow_versions: Counter[str] = Counter()
    infrastructure_statuses: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    obligation_categories: Counter[str] = Counter()
    assessment_statuses: Counter[str] = Counter()
    obligation_source_statuses: Counter[str] = Counter()
    criterion_source_checks: Counter[str] = Counter()
    challenge_kinds: Counter[str] = Counter()
    challenge_dispositions: Counter[str] = Counter()
    repair_audit_statuses: Counter[str] = Counter()
    repair_sequence_observation_statuses: Counter[str] = Counter()
    repair_source_checks: Counter[str] = Counter()
    repair_decisions: Counter[str] = Counter()
    role_runs: Counter[str] = Counter()
    role_timeouts: Counter[str] = Counter()
    role_wall_time_sec: Counter[str] = Counter()
    stage_a_first_valid = stage_a_normalized_valid = stage_a_final_valid = 0
    stage_b_first_valid = stage_b_normalized_valid = stage_b_final_valid = 0
    stage_a_attempted = stage_b_attempted = 0
    approve_failures = approve_count = 0
    metric_approve_count = metric_abstain_count = 0
    source_anchor_attempted = source_anchor_valid = 0
    effective_criteria = unresolved_criteria = 0
    effective_obligations = nonbinding_obligations = unresolved_obligations = 0
    audited_requests = implemented_requests = unaddressed_requests = 0
    evaluated = recovered = run_evaluated = 0
    evaluated_baseline_failures = evaluated_baseline_passes = 0
    baseline_failures = baseline_passes = p1_passes = 0
    details: dict[str, Any] = {}
    patch_size_changes: list[int] = []
    late_obligations = unexpected_regressions = repair_requests = 0
    f2p_recovered = p2p_regressed = 0

    for task_id in manifest.task_ids:
        is_metric_task = task_id in metric_task_ids
        baseline_trial = manifest.source_job_dir / manifest.patches[task_id].source_trial_id
        baseline_rewards = _load_json(baseline_trial / "verifier" / "reward.json") or {}
        baseline_reward = baseline_rewards.get("reward")
        if is_metric_task and isinstance(baseline_reward, (int, float)):
            if float(baseline_reward) == 1.0:
                baseline_passes += 1
            else:
                baseline_failures += 1
        trials = trials_by_task.get(task_id, [])
        if not trials:
            target_transitions = transitions if is_metric_task else excluded_transitions
            target_transitions["pending"] += 1
            details[task_id] = {
                "transition": "pending",
                "included_in_primary_metrics": is_metric_task,
                "p0_rewards": baseline_rewards,
            }
            continue
        latest = max(
            trials,
            key=lambda value: str(value.get("finished_at") or value.get("started_at") or ""),
        )
        trial_dir = Path(latest["_trial_dir"])
        verifier = latest.get("verifier_result") or {}
        rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
        p1_reward = rewards.get("reward") if isinstance(rewards, dict) else None
        legacy_workflow_root = trial_dir / "review_repair"
        v5_workflow_root = trial_dir / "v5_review_repair"
        legacy_manifest = _load_json(legacy_workflow_root / "manifest.json")
        v5_manifest = _load_json(v5_workflow_root / "manifest.json")
        is_v5 = legacy_manifest is None and v5_manifest is not None
        workflow_root = v5_workflow_root if is_v5 else legacy_workflow_root
        workflow_manifest = v5_manifest if is_v5 else legacy_manifest
        manifest_payload = workflow_manifest or {}
        contract_version = manifest_payload.get("contract_version") or "unknown"
        prompt_version = manifest_payload.get("prompt_version") or "unknown"
        workflow_versions[f"contract={contract_version};prompt={prompt_version}"] += 1
        ledger = _load_json(
            workflow_root / ("v5/ledger.json" if is_v5 else "reviewer/review_obligations.json")
        )
        assessment = (
            None
            if is_v5
            else _load_json(workflow_root / "reviewer" / "patch_assessment.json")
        )
        stage_a_validation = _load_json(workflow_root / "reviewer" / "stage_a_validation.json")
        stage_b_validation = _load_json(workflow_root / "reviewer" / "stage_b_validation.json")
        source_validation = _load_json(workflow_root / "reviewer" / "source_anchor_validation.json")
        effective_set = _load_json(workflow_root / "reviewer" / "effective_review_set.json")
        repair_audit_validation = _load_json(
            workflow_root / "repair" / "repair_audit_validation.json"
        )
        repair_audit = _load_json(workflow_root / "repair" / "repair_audit.json")
        resource_usage = _load_json(workflow_root / "outputs" / "resource_usage.json")
        verifier_observability = _verifier_observability(trial_dir)

        if stage_a_validation is not None:
            stage_a_attempted += 1
            if stage_a_validation.get("valid") is True:
                stage_a_final_valid += 1
                if stage_a_validation.get("correction_used") is False:
                    stage_a_normalized_valid += 1
                    if stage_a_validation.get("normalization_used") is not True:
                        stage_a_first_valid += 1
        if stage_b_validation is not None:
            stage_b_attempted += 1
            if stage_b_validation.get("valid") is True:
                stage_b_final_valid += 1
                if stage_b_validation.get("correction_used") is False:
                    stage_b_normalized_valid += 1
                    if stage_b_validation.get("normalization_used") is not True:
                        stage_b_first_valid += 1
        if source_validation is not None:
            source_anchor_attempted += 1
            if source_validation.get("valid") is True:
                source_anchor_valid += 1

        state = (
            manifest_payload.get("workflow_state")
            if is_v5
            else manifest_payload.get("final_state")
        )
        if isinstance(state, str):
            workflow_states[state] += 1
        route = manifest_payload.get("route")
        if isinstance(route, str):
            workflow_routes[route] += 1
        verdict = manifest_payload.get("reviewer_verdict")
        if isinstance(verdict, str):
            verdicts[verdict] += 1
            if is_metric_task and verdict == "approve":
                metric_approve_count += 1
            if is_metric_task and verdict == "abstain":
                metric_abstain_count += 1
        repair_audit_status = manifest_payload.get("repair_audit_status")
        if isinstance(repair_audit_status, str):
            repair_audit_statuses[repair_audit_status] += 1
        repair_sequence_status = manifest_payload.get("repair_sequence_observation_status")
        if isinstance(repair_sequence_status, str):
            repair_sequence_observation_statuses[repair_sequence_status] += 1
        if ledger is not None:
            for obligation in ledger.get("obligations") or []:
                category = obligation.get("category") if isinstance(obligation, dict) else None
                if isinstance(category, str):
                    obligation_categories[category] += 1
        if assessment is not None:
            for item in assessment.get("obligation_assessments") or []:
                status = (
                    item.get("implementation_status", item.get("status"))
                    if isinstance(item, dict)
                    else None
                )
                if isinstance(status, str):
                    assessment_statuses[status] += 1
                source_status = item.get("source_status") if isinstance(item, dict) else None
                if isinstance(source_status, str):
                    obligation_source_statuses[source_status] += 1
            for item in assessment.get("criterion_assessments") or []:
                source_check = item.get("source_check") if isinstance(item, dict) else None
                status = source_check.get("status") if isinstance(source_check, dict) else None
                if isinstance(status, str):
                    criterion_source_checks[status] += 1
            for item in assessment.get("ledger_challenges") or []:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("kind"), str):
                    challenge_kinds[item["kind"]] += 1
                if isinstance(item.get("disposition"), str):
                    challenge_dispositions[item["disposition"]] += 1
            late_obligations += len(
                assessment.get("late_obligations")
                or assessment.get("late_obligation_candidates")
                or []
            )
            unexpected_regressions += len(assessment.get("unexpected_regressions") or [])
            repair_requests += len(assessment.get("repair_requests") or [])
        if effective_set is not None:
            effective_criteria += len(effective_set.get("effective_criteria") or [])
            unresolved_criteria += len(effective_set.get("unresolved_criteria") or [])
            effective_obligations += len(effective_set.get("effective_obligations") or [])
            nonbinding_obligations += len(effective_set.get("nonbinding_obligations") or [])
            unresolved_obligations += len(effective_set.get("unresolved_obligations") or [])
        if repair_audit is not None:
            request_ids = (
                set(effective_set.get("repair_request_ids") or []) if effective_set else set()
            )
            audited_requests += len(request_ids)
            implemented: set[str] = set()
            for item in repair_audit.get("changes") or []:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("source_check"), str):
                    repair_source_checks[item["source_check"]] += 1
                if isinstance(item.get("decision"), str):
                    repair_decisions[item["decision"]] += 1
                if item.get("decision") == "implement":
                    implemented.update(item.get("request_refs") or [])
            implemented_requests += len(implemented & request_ids)
            unaddressed_requests += len(repair_audit.get("unaddressed_requests") or [])
        if resource_usage is not None:
            for item in resource_usage.get("roles") or []:
                if not isinstance(item, dict) or not isinstance(item.get("role"), str):
                    continue
                role = item["role"]
                role_runs[role] += 1
                if item.get("timed_out") is True:
                    role_timeouts[role] += 1
                usage = item.get("resource_usage") or {}
                wall_time = usage.get("wall_time_sec") if isinstance(usage, dict) else None
                if isinstance(wall_time, (int, float)):
                    role_wall_time_sec[role] += wall_time

        p0_path = workflow_root / "outputs" / (
            "v5-p0.patch" if is_v5 else "p0.patch"
        )
        if not is_v5:
            p0_path = workflow_root / "inputs" / "p0.patch"
        p1_path = workflow_root / "outputs" / (
            "v5-p1.patch" if is_v5 else "p1.patch"
        )
        pairing_errors = _published_p1_errors(
            workflow_manifest=workflow_manifest,
            p1_path=p1_path,
            p1_reward=p1_reward,
        )
        valid_pair = not pairing_errors
        patch_delta = None
        if valid_pair and p0_path.is_file():
            patch_delta = p1_path.stat().st_size - p0_path.stat().st_size
            if is_metric_task:
                patch_size_changes.append(patch_delta)

        transition = "infrastructure_error"
        if (
            valid_pair
            and isinstance(baseline_reward, (int, float))
            and isinstance(p1_reward, (int, float))
        ):
            run_evaluated += 1
            if is_metric_task:
                evaluated += 1
                if float(p1_reward) == 1.0:
                    p1_passes += 1
                if float(baseline_reward) == 1.0:
                    evaluated_baseline_passes += 1
                else:
                    evaluated_baseline_failures += 1
            if float(baseline_reward) == 1.0:
                transition = "pass_to_pass" if float(p1_reward) == 1.0 else "pass_to_fail"
            elif float(p1_reward) == 1.0:
                transition = "fail_to_pass"
                if is_metric_task:
                    recovered += 1
            else:
                baseline_partial = baseline_rewards.get("partial")
                p1_partial = rewards.get("partial") if isinstance(rewards, dict) else None
                if isinstance(baseline_partial, (int, float)) and isinstance(
                    p1_partial, (int, float)
                ):
                    if p1_partial > baseline_partial:
                        transition = "fail_to_fail_score_improved"
                    elif p1_partial < baseline_partial:
                        transition = "fail_to_fail_score_declined"
                    else:
                        transition = "fail_to_fail_score_unchanged"
                else:
                    transition = "fail_to_fail_score_unknown"
        infrastructure = _infrastructure_failure_metadata(
            trial_dir=trial_dir,
            result=latest,
            transition=transition,
        )
        if infrastructure["infrastructure_status"] is not None:
            infrastructure_statuses[infrastructure["infrastructure_status"]] += 1
        if infrastructure["failure_class"] is not None:
            failure_classes[infrastructure["failure_class"]] += 1
        target_transitions = transitions if is_metric_task else excluded_transitions
        target_transitions[transition] += 1
        baseline_f2p = baseline_rewards.get("f2p_passed")
        p1_f2p = rewards.get("f2p_passed") if isinstance(rewards, dict) else None
        if (
            valid_pair
            and is_metric_task
            and isinstance(baseline_f2p, int)
            and isinstance(p1_f2p, int)
        ):
            f2p_recovered += max(p1_f2p - baseline_f2p, 0)
        baseline_p2p = baseline_rewards.get("p2p_passed")
        p1_p2p = rewards.get("p2p_passed") if isinstance(rewards, dict) else None
        if (
            valid_pair
            and is_metric_task
            and isinstance(baseline_p2p, int)
            and isinstance(p1_p2p, int)
        ):
            p2p_regressed += max(baseline_p2p - p1_p2p, 0)
        if (
            valid_pair
            and is_metric_task
            and verdict == "approve"
            and isinstance(p1_reward, (int, float))
        ):
            approve_count += 1
            if float(p1_reward) != 1.0:
                approve_failures += 1
        details[task_id] = {
            "transition": transition,
            "included_in_primary_metrics": is_metric_task,
            "p0_rewards": baseline_rewards,
            "p1_rewards": rewards,
            "workflow_state": state,
            "workflow_route": route,
            "verdict": verdict,
            "repair_executed": manifest_payload.get("repair_executed"),
            "contract_version": contract_version,
            "prompt_version": prompt_version,
            "experiment_arm": manifest_payload.get("experiment_arm"),
            "abstain_reason": manifest_payload.get("abstain_reason"),
            "repair_audit_status": repair_audit_status,
            "repair_sequence_observation_status": repair_sequence_status,
            "source_anchor_valid": (
                source_validation.get("valid") if source_validation is not None else None
            ),
            "effective_criterion_count": (
                len(effective_set.get("effective_criteria") or [])
                if effective_set is not None
                else None
            ),
            "unresolved_criterion_count": (
                len(effective_set.get("unresolved_criteria") or [])
                if effective_set is not None
                else None
            ),
            "effective_obligation_count": (
                len(effective_set.get("effective_obligations") or [])
                if effective_set is not None
                else None
            ),
            "nonbinding_obligation_count": (
                len(effective_set.get("nonbinding_obligations") or [])
                if effective_set is not None
                else None
            ),
            "unresolved_obligation_count": (
                len(effective_set.get("unresolved_obligations") or [])
                if effective_set is not None
                else None
            ),
            "late_obligation_count": (
                len(
                    assessment.get("late_obligations")
                    or assessment.get("late_obligation_candidates")
                    or []
                )
                if assessment is not None
                else None
            ),
            "repair_audit_valid": (
                repair_audit_validation.get("valid")
                if repair_audit_validation is not None
                else None
            ),
            "p0_sha256": manifest_payload.get("p0_sha256"),
            "p1_sha256": manifest_payload.get("p1_sha256"),
            "valid_p1_pair": valid_pair,
            "pairing_errors": pairing_errors,
            **infrastructure,
            **verifier_observability,
            "patch_size_delta_bytes": patch_delta,
            "trial_dir": str(trial_dir),
        }

    denominator = len(metric_task_ids)
    all_metric_tasks_evaluated = evaluated == denominator
    near_miss_only = baseline_failures == denominator and baseline_passes == 0
    pass_to_pass = transitions.get("pass_to_pass", 0)
    return {
        "benchmark": "deepswe",
        "benchmark_revision": "1.1",
        "experiment": "specification-reviewer-repair-once",
        "sample_characterization": (
            "exploratory_near_miss_baseline_failures"
            if near_miss_only
            else "paired_random60_p0_to_p1_excluding_development_cases"
        ),
        "metric_name": (
            "recovery_rate_not_overall_pass_rate_uplift"
            if near_miss_only
            else "paired_p0_to_p1_pass_rate"
        ),
        "job_dir": str(root),
        "source_job_id": manifest.source_job_id,
        "task_count": len(manifest.task_ids),
        "metric_task_count": denominator,
        "excluded_task_count": len(excluded_task_ids),
        "excluded_task_ids": sorted(excluded_task_ids),
        "run_evaluated_count": run_evaluated,
        "evaluated_count": evaluated,
        "p0_passed_count": baseline_passes,
        "p1_passed_count": p1_passes,
        "p0_pass_rate": (
            baseline_passes / denominator
            if baseline_passes + baseline_failures == denominator
            else None
        ),
        "p1_pass_rate": p1_passes / denominator if all_metric_tasks_evaluated else None,
        "net_pass_change": p1_passes - baseline_passes if all_metric_tasks_evaluated else None,
        "recovered_count": recovered,
        "recovery_rate": (
            recovered / baseline_failures
            if all_metric_tasks_evaluated and baseline_failures
            else None
        ),
        "observed_recovery_rate": _rate(recovered, evaluated_baseline_failures),
        "retention_rate": _rate(pass_to_pass, evaluated_baseline_passes),
        "transitions": dict(sorted(transitions.items())),
        "excluded_transitions": dict(sorted(excluded_transitions.items())),
        "workflow_states": dict(sorted(workflow_states.items())),
        "workflow_routes": dict(sorted(workflow_routes.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "workflow_versions": dict(sorted(workflow_versions.items())),
        "infrastructure_statuses": dict(sorted(infrastructure_statuses.items())),
        "failure_classes": dict(sorted(failure_classes.items())),
        "schema_validity": {
            "stage_a_first_attempt": _rate(stage_a_first_valid, stage_a_attempted),
            "stage_a_after_normalization": _rate(stage_a_normalized_valid, stage_a_attempted),
            "stage_a_after_correction": _rate(stage_a_final_valid, stage_a_attempted),
            "stage_b_first_attempt": _rate(stage_b_first_valid, stage_b_attempted),
            "stage_b_after_normalization": _rate(stage_b_normalized_valid, stage_b_attempted),
            "stage_b_after_correction": _rate(stage_b_final_valid, stage_b_attempted),
        },
        "false_approve_rate": _rate(approve_failures, approve_count),
        "approval_coverage": _rate(metric_approve_count, denominator),
        "abstain_rate": _rate(metric_abstain_count, denominator),
        "source_anchor_validity": _rate(source_anchor_valid, source_anchor_attempted),
        "obligation_categories": dict(sorted(obligation_categories.items())),
        "assessment_statuses": dict(sorted(assessment_statuses.items())),
        "obligation_source_statuses": dict(sorted(obligation_source_statuses.items())),
        "criterion_source_checks": dict(sorted(criterion_source_checks.items())),
        "ledger_challenge_kinds": dict(sorted(challenge_kinds.items())),
        "ledger_challenge_dispositions": dict(sorted(challenge_dispositions.items())),
        "effective_criterion_count": effective_criteria,
        "unresolved_criterion_count": unresolved_criteria,
        "effective_obligation_count": effective_obligations,
        "nonbinding_obligation_count": nonbinding_obligations,
        "unresolved_obligation_count": unresolved_obligations,
        "repair_audit_statuses": dict(sorted(repair_audit_statuses.items())),
        "repair_sequence_observation_statuses": dict(
            sorted(repair_sequence_observation_statuses.items())
        ),
        "repair_source_checks": dict(sorted(repair_source_checks.items())),
        "repair_decisions": dict(sorted(repair_decisions.items())),
        "audited_request_count": audited_requests,
        "implemented_request_count": implemented_requests,
        "unaddressed_request_count": unaddressed_requests,
        "audited_request_closure": _rate(implemented_requests, audited_requests),
        "role_resources": {
            role: {
                "runs": role_runs[role],
                "timed_out": role_timeouts[role],
                "wall_time_sec": role_wall_time_sec[role],
            }
            for role in sorted(role_runs)
        },
        "late_obligation_count": late_obligations,
        "stage_a_omission_count": late_obligations,
        "unexpected_regression_count": unexpected_regressions,
        "repair_request_count": repair_requests,
        "f2p_recovered_count": f2p_recovered,
        "p2p_regressed_count": p2p_regressed,
        "mean_patch_size_delta_bytes": (
            sum(patch_size_changes) / len(patch_size_changes) if patch_size_changes else None
        ),
        "details": details,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
