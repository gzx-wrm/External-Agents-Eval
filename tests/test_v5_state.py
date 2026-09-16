from __future__ import annotations

import json

import pytest

from external_agents_eval.workflows.solve_review_repair.v5_contracts import (
    V5ContractError,
    validate_review_note,
    validate_summary_output,
)
from external_agents_eval.workflows.solve_review_repair.v5_state import (
    ImmutableBundleStore,
    V5StateStore,
)

NOTE = """# Review note: O001

## Summary

The target was reviewed from the public entry point.

## Review steps

1. **Static inspection**

Inspected the relevant source and recorded the observed behavior.

## Limits and conflicting observations

No additional limitation was identified.

## Reproduction materials

No replay material was preserved for this static review.
"""


def batch(*, status="satisfied", target_id="O001", note="notes/O001.md", done=False):
    return {
        "new_targets": [],
        "assessments": [{
            "target_id": target_id,
            "status": status,
            "expected": "The public behavior is preserved.",
            "observed": "The public behavior is preserved.",
            "reason": "The source and observed behavior agree.",
            "code_refs": [],
            "review_note_path": note,
        }],
        "revisit_target_ids": [],
        "done": done,
    }


def stage(store, key, payload, notes=None):
    return store.stage(
        batch_key=key,
        normalized_json=payload,
        raw_output=json.dumps(payload),
        notes={"notes/O001.md": NOTE.encode()} if notes is None else notes,
        trace_manifest={"segments": []},
        materials_manifest={"missing": []},
    )


def test_note_validator_rejects_escape_and_placeholder(tmp_path):
    root = tmp_path / "attempt"
    (root / "notes").mkdir(parents=True)
    (root / "notes/O001.md").write_text(NOTE, encoding="utf-8")
    assert validate_review_note(root, "notes/O001.md", "O001").startswith(b"# Review note")
    with pytest.raises(V5ContractError, match="escapes"):
        validate_review_note(root, "../O001.md", "O001")
    (root / "notes/O001.md").write_text(
        NOTE.replace("No replay", "{{TARGET_ID}}"), encoding="utf-8"
    )
    with pytest.raises(V5ContractError, match="placeholder"):
        validate_review_note(root, "notes/O001.md", "O001")
    (root / "notes/O001.md").write_text(
        NOTE.replace(
            "Inspected the relevant source and recorded the observed behavior.",
            "Inspected a template containing {{ user_name }} in the relevant source.",
        ),
        encoding="utf-8",
    )
    assert validate_review_note(root, "notes/O001.md", "O001")
    (root / "notes/O001.md").write_text(
        NOTE.replace(
            "The target was reviewed from the public entry point.", ""
        ),
        encoding="utf-8",
    )
    with pytest.raises(V5ContractError, match="empty section"):
        validate_review_note(root, "notes/O001.md", "O001")
    (root / "real.md").write_text(NOTE, encoding="utf-8")
    (root / "notes/O001.md").unlink()
    (root / "notes/O001.md").symlink_to(root / "real.md")
    with pytest.raises(V5ContractError, match="regular file"):
        validate_review_note(root, "notes/O001.md", "O001")


def test_atomic_commit_replay_and_changed_content_conflict(tmp_path):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    state.create_run("run", {"task": "x"})
    root = tmp_path / "attempt-1"
    (root / "notes").mkdir(parents=True)
    (root / "notes/O001.md").write_text(NOTE, encoding="utf-8")
    payload = batch()
    state.prepare_attempt("run", "I", 1, 1, input_payload={"round": 1})
    bundle = stage(bundles, "run:I:1:1", payload)
    first = state.commit_batch("run", "I", 1, 1, payload, output_root=root, bundle=bundle)
    replay = state.commit_batch("run", "I", 1, 1, payload, output_root=root, bundle=bundle)
    assert replay == first
    changed = batch(status="violated")
    with pytest.raises(V5ContractError, match="different content"):
        state.commit_batch("run", "I", 1, 1, changed, output_root=root, bundle=bundle)
    assert state.lane("run", "I")["effective"]["O001"]["version"] == 1


def test_changed_note_content_cannot_overwrite_a_committed_batch(tmp_path):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    state.create_run("run", {"task": "x"})
    root = tmp_path / "attempt-1"
    (root / "notes").mkdir(parents=True)
    (root / "notes/O001.md").write_text(NOTE, encoding="utf-8")
    payload = batch()
    state.prepare_attempt("run", "I", 1, 1, input_payload={"round": 1})
    first = stage(bundles, "first-note", payload)
    state.commit_batch(
        "run", "I", 1, 1, payload, output_root=root, bundle=first
    )

    changed_note = NOTE.replace(
        "The target was reviewed from the public entry point.",
        "The target was reviewed from a different entry point.",
    ).encode()
    replacement = stage(
        bundles,
        "changed-note",
        payload,
        notes={"notes/O001.md": changed_note},
    )
    with pytest.raises(V5ContractError, match="different content"):
        state.commit_batch(
            "run", "I", 1, 1, payload, output_root=root, bundle=replacement
        )


def test_invalid_batch_and_stale_attempt_have_no_partial_state_effect(tmp_path):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    state.create_run("run", {"task": "x"})
    state.initialize_lanes(
        "run",
        implementation_targets=["O001"],
        regression_targets=[],
        target_catalog={
            "O001": {
                "statement": "Initial",
                "importance": "required",
                "sources": [],
            }
        },
    )
    missing_root = tmp_path / "missing-note"
    missing_root.mkdir()
    invalid = batch()
    invalid["new_targets"] = [
        {
            "statement": "A new adjacent target",
            "importance": "advisory",
            "sources": [
                {"kind": "task", "field": "description", "quote": "target"}
            ],
        }
    ]
    state.prepare_attempt("run", "I", 1, 1, input_payload={"attempt": 1})
    before = state.lane("run", "I")
    with pytest.raises(V5ContractError, match="not a regular file"):
        state.commit_batch(
            "run",
            "I",
            1,
            1,
            invalid,
            output_root=missing_root,
            bundle=stage(bundles, "missing-note", invalid, notes={}),
        )
    assert state.lane("run", "I") == before

    valid_root = tmp_path / "valid-note"
    (valid_root / "notes").mkdir(parents=True)
    (valid_root / "notes/O001.md").write_text(NOTE, encoding="utf-8")
    state.prepare_attempt("run", "I", 1, 2, input_payload={"attempt": 2})
    state.prepare_attempt("run", "I", 1, 3, input_payload={"attempt": 3})
    valid = batch()
    state.commit_batch(
        "run",
        "I",
        1,
        2,
        valid,
        output_root=valid_root,
        bundle=stage(bundles, "valid", valid),
    )
    committed = state.lane("run", "I")
    with pytest.raises(V5ContractError, match="stale"):
        state.commit_batch(
            "run",
            "I",
            1,
            3,
            valid,
            output_root=valid_root,
            bundle=stage(bundles, "stale", valid),
        )
    assert state.lane("run", "I") == committed


def test_reopen_and_replacement_version_only_changes_on_valid_commit(tmp_path):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    state.create_run("run", {"task": "x"})
    root = tmp_path / "attempt-1"
    (root / "notes").mkdir(parents=True)
    (root / "notes/O001.md").write_text(NOTE, encoding="utf-8")
    first = batch()
    state.prepare_attempt("run", "I", 1, 1, input_payload={})
    state.commit_batch("run", "I", 1, 1, first, output_root=root, bundle=stage(bundles, "a", first))
    reopen = {"new_targets": [], "assessments": [], "revisit_target_ids": ["O001"], "done": False}
    state.prepare_attempt("run", "I", 2, 1, input_payload={"reopen": True})
    state.commit_batch("run", "I", 2, 1, reopen, output_root=root, bundle=stage(bundles, "b", reopen, notes={}))
    assert state.lane("run", "I")["effective"] == {}
    state.prepare_attempt("run", "I", 3, 1, input_payload={"replacement": True})
    replacement = batch(status="unresolved")
    state.commit_batch("run", "I", 3, 1, replacement, output_root=root, bundle=stage(bundles, "c", replacement))
    assert state.lane("run", "I")["effective"]["O001"]["version"] == 2
    assert len(state.lane("run", "I")["history"]["O001"]) == 2

    state.prepare_attempt("run", "I", 4, 1, input_payload={"reopen_again": True})
    with pytest.raises(V5ContractError, match="already been reopened once"):
        state.commit_batch(
            "run",
            "I",
            4,
            1,
            reopen,
            output_root=root,
            bundle=stage(bundles, "reopen-again", reopen, notes={}),
        )


def test_done_and_summary_freeze_are_durable(tmp_path):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    state.create_run("run", {"task": "x"})
    root = tmp_path / "attempt-1"
    (root / "notes").mkdir(parents=True)
    (root / "notes/O001.md").write_text(NOTE, encoding="utf-8")
    payload = batch(done=True)
    state.prepare_attempt("run", "I", 1, 1, input_payload={})
    state.commit_batch("run", "I", 1, 1, payload, output_root=root, bundle=stage(bundles, "done", payload))
    state.stop_lane("run", "R", reason="test-stop")
    snapshot = {"I": state.lane("run", "I"), "R": state.lane("run", "R")}
    freeze = state.freeze_summary("run", snapshot)
    assert freeze
    assert state.freeze_summary("run", snapshot) == freeze
    with pytest.raises(V5ContractError, match="different snapshot"):
        state.freeze_summary("run", {})

    prepared = state.prepare_delivery(
        "run",
        freeze_id=freeze,
        route="repair",
        repair_invocation_id="run.repair",
    )
    assert prepared["status"] == "PREPARED"
    assert state.prepare_delivery(
        "run",
        freeze_id=freeze,
        route="repair",
        repair_invocation_id="run.repair",
    )["status"] == "PREPARED"
    with pytest.raises(V5ContractError, match="different inputs"):
        state.prepare_delivery(
            "run",
            freeze_id=freeze,
            route="hold",
            repair_invocation_id=None,
        )
    result = {"route": "repair", "repair_executed": True}
    state.finish_delivery("run", p1_sha256="a" * 64, result=result)
    state.finish_delivery("run", p1_sha256="a" * 64, result=result)
    assert state.delivery("run")["status"] == "COMPLETED"
    assert state.run_record("run")["phase"] == "completed"
    with pytest.raises(V5ContractError, match="different output"):
        state.finish_delivery(
            "run", p1_sha256="b" * 64, result=result
        )


def test_run_attempt_and_invocation_identities_are_idempotent(tmp_path):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    assert state.create_run("run", {"task": "x"}) is True
    assert state.create_run("run", {"task": "x"}) is False
    with pytest.raises(V5ContractError, match="different inputs"):
        state.create_run("run", {"task": "y"})

    state.initialize_lanes(
        "run",
        implementation_targets=["O001"],
        regression_targets=["P001"],
        target_catalog={
            "O001": {"statement": "x", "importance": "required", "sources": []},
            "P001": {"statement": "y", "importance": "required", "sources": []},
        },
    )
    lane_deadline = state.lane("run", "I")["deadline_at"]
    state.initialize_lanes(
        "run",
        implementation_targets=["O001"],
        regression_targets=["P001"],
        target_catalog={
            "O001": {"statement": "x", "importance": "required", "sources": []},
            "P001": {"statement": "y", "importance": "required", "sources": []},
        },
    )
    assert state.lane("run", "I")["deadline_at"] == lane_deadline
    assert state.prepare_attempt("run", "I", 1, 1, input_payload={"round": 1})
    assert state.prepare_attempt("run", "I", 1, 1, input_payload={"round": 1})
    with pytest.raises(V5ContractError, match="different input"):
        state.prepare_attempt("run", "I", 1, 1, input_payload={"round": 2})

    assert state.prepare_invocation(
        "invocation",
        run_id="run",
        role="v5_implementation_reviewer",
        prompt_sha256="a" * 64,
        worktree="candidate",
    )
    assert not state.prepare_invocation(
        "invocation",
        run_id="run",
        role="v5_implementation_reviewer",
        prompt_sha256="a" * 64,
        worktree="candidate",
    )
    result = {"exit_code": 0, "timed_out": False, "exception": None}
    state.finish_invocation("invocation", status="COMPLETED", result=result)
    state.finish_invocation("invocation", status="COMPLETED", result=result)
    assert state.invocation("invocation")["status"] == "COMPLETED"
    with pytest.raises(V5ContractError, match="different result"):
        state.finish_invocation(
            "invocation",
            status="FAILED",
            result={"exit_code": 1, "timed_out": False, "exception": "failed"},
        )

    deadline = state.stage_deadline("run", "A", duration_sec=120)
    assert state.stage_deadline("run", "A", duration_sec=120) == deadline
    with pytest.raises(V5ContractError, match="another duration"):
        state.stage_deadline("run", "A", duration_sec=121)


def test_new_target_is_registered_once_and_available_next_round(tmp_path):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    state.create_run("run", {"task": "x"})
    state.initialize_lanes(
        "run",
        implementation_targets=["O001"],
        regression_targets=[],
        target_catalog={
            "O001": {"statement": "Initial", "importance": "required", "sources": []}
        },
    )
    payload = {
        "new_targets": [
            {
                "statement": "Adjacent behavior",
                "importance": "advisory",
                "sources": [
                    {"kind": "task", "field": "description", "quote": "behavior"}
                ],
            }
        ],
        "assessments": [],
        "revisit_target_ids": [],
        "done": False,
    }
    root = tmp_path / "attempt"
    root.mkdir()
    state.prepare_attempt("run", "I", 1, 1, input_payload={})
    receipt = state.commit_batch(
        "run",
        "I",
        1,
        1,
        payload,
        output_root=root,
        bundle=stage(bundles, "new-target", payload, notes={}),
    )
    assert receipt["assigned_target_ids"] == ["I001"]
    lane = state.lane("run", "I")
    assert lane["targets"]["I001"]["statement"] == "Adjacent behavior"
    assert lane["pending"] == ["I001", "O001"]


def test_immutable_bundle_replay_rejects_corrupted_existing_file(tmp_path):
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    payload = batch()
    original = stage(bundles, "same-batch", payload)
    (original.path / "notes/O001.md").write_text("corrupt", encoding="utf-8")
    with pytest.raises(V5ContractError, match="corrupt"):
        stage(bundles, "same-batch", payload)


def test_immutable_bundle_refuses_unredacted_secret(tmp_path):
    bundles = ImmutableBundleStore(tmp_path / "artifacts", secrets=("secret-token",))
    payload = batch()

    with pytest.raises(V5ContractError, match="unredacted secret"):
        bundles.stage(
            batch_key="secret-batch",
            normalized_json=payload,
            raw_output="secret-token",
            notes={"notes/O001.md": NOTE.encode()},
            trace_manifest={"segments": []},
            materials_manifest={"missing": []},
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (batch(target_id="unknown"), "unknown target"),
        (
            {
                **batch(),
                "assessments": [
                    *batch()["assessments"],
                    *batch()["assessments"],
                ],
            },
            "same target twice",
        ),
        (
            {
                "new_targets": [],
                "assessments": [],
                "revisit_target_ids": [],
                "done": True,
            },
            "done=true requires no pending target",
        ),
    ],
)
def test_commit_rejects_unknown_duplicate_and_premature_done(
    tmp_path, payload, message
):
    state = V5StateStore(tmp_path / "workflow.sqlite")
    bundles = ImmutableBundleStore(tmp_path / "artifacts")
    state.create_run("run", {"task": "x"})
    state.initialize_lanes(
        "run",
        implementation_targets=["O001"],
        regression_targets=[],
        target_catalog={
            "O001": {"statement": "Initial", "importance": "required", "sources": []}
        },
    )
    root = tmp_path / "attempt"
    (root / "notes").mkdir(parents=True)
    (root / "notes/O001.md").write_text(NOTE, encoding="utf-8")
    state.prepare_attempt("run", "I", 1, 1, input_payload={})
    bundle = stage(
        bundles,
        "invalid-batch",
        payload,
        notes=(
            {"notes/O001.md": NOTE.encode()}
            if payload.get("assessments")
            else {}
        ),
    )

    with pytest.raises(V5ContractError, match=message):
        state.commit_batch(
            "run", "I", 1, 1, payload, output_root=root, bundle=bundle
        )


def test_summary_cannot_route_advisory_or_unresolved_finding_to_repair():
    source = {"kind": "task", "field": "description", "quote": "behavior"}
    payload = {
        "summary": "Repair requested.",
        "decisions": [
            {
                "finding_ref": "I/I001",
                "disposition": "repair",
                "reason": "The reviewer requested a change.",
                "supporting_refs": ["I/I001"],
                "sources": [source],
            }
        ],
        "repair_requests": [
            {
                "finding_refs": ["I/I001"],
                "change": "Change the adjacent behavior.",
                "validation_record_refs": ["I/I001"],
                "preserve_refs": [],
            }
        ],
        "new_questions": [],
    }

    for context in (
        {
            "I/I001": {
                "kind": "assessment",
                "status": "violated",
                "importance": "advisory",
            }
        },
        {
            "I/I001": {
                "kind": "assessment",
                "status": "unresolved",
                "importance": "required",
            }
        },
    ):
        with pytest.raises(V5ContractError, match="required violated"):
            validate_summary_output(
                payload,
                {"I/I001"},
                decision_context=context,
                allowed_sources=[source],
            )


def test_summary_may_challenge_a_satisfied_finding_only_as_unresolved():
    source = {"kind": "task", "field": "description", "quote": "behavior"}
    context = {
        "I/O001": {
            "kind": "assessment",
            "status": "satisfied",
            "importance": "required",
        }
    }
    decision = {
        "finding_ref": "I/O001",
        "disposition": "unresolved",
        "reason": "The archived observations conflict.",
        "supporting_refs": ["I/O001"],
        "sources": [source],
    }
    payload = {
        "summary": "The apparent success remains uncertain.",
        "decisions": [decision],
        "repair_requests": [],
        "new_questions": [],
    }

    assert validate_summary_output(
        payload,
        set(),
        decision_context=context,
        allowed_sources=[source],
    ) == payload
    decision["disposition"] = "dismiss"
    with pytest.raises(V5ContractError, match="invalid extra"):
        validate_summary_output(
            payload,
            set(),
            decision_context=context,
            allowed_sources=[source],
        )


def test_summary_validation_records_cannot_reference_a_question():
    source = {"kind": "task", "field": "description", "quote": "behavior"}
    context = {
        "A/Q001": {"kind": "question", "blocking": True},
        "I/O001": {
            "kind": "assessment",
            "status": "violated",
            "importance": "required",
        },
    }
    payload = {
        "summary": "One repair is required.",
        "decisions": [
            {
                "finding_ref": "A/Q001",
                "disposition": "unresolved",
                "reason": "The question remains open.",
                "supporting_refs": ["A/Q001"],
                "sources": [source],
            },
            {
                "finding_ref": "I/O001",
                "disposition": "repair",
                "reason": "The required behavior is violated.",
                "supporting_refs": ["I/O001"],
                "sources": [source],
            },
        ],
        "repair_requests": [
            {
                "finding_refs": ["I/O001"],
                "change": "Implement the required behavior.",
                "validation_record_refs": ["A/Q001"],
                "preserve_refs": [],
            }
        ],
        "new_questions": [],
    }

    with pytest.raises(V5ContractError, match="effective assessments"):
        validate_summary_output(
            payload,
            {"A/Q001", "I/O001"},
            decision_context=context,
            allowed_sources=[source],
        )


def test_summary_cannot_preserve_a_satisfied_finding_it_challenges():
    source = {"kind": "task", "field": "description", "quote": "behavior"}
    context = {
        "I/O001": {
            "kind": "assessment",
            "status": "violated",
            "importance": "required",
        },
        "R/P001": {
            "kind": "assessment",
            "status": "satisfied",
            "importance": "required",
        },
    }
    payload = {
        "summary": "One repair is required and one prior success is disputed.",
        "decisions": [
            {
                "finding_ref": "I/O001",
                "disposition": "repair",
                "reason": "The required behavior is violated.",
                "supporting_refs": ["I/O001"],
                "sources": [source],
            },
            {
                "finding_ref": "R/P001",
                "disposition": "unresolved",
                "reason": "The archived observations conflict.",
                "supporting_refs": ["R/P001"],
                "sources": [source],
            },
        ],
        "repair_requests": [
            {
                "finding_refs": ["I/O001"],
                "change": "Implement the required behavior.",
                "validation_record_refs": ["I/O001"],
                "preserve_refs": ["R/P001"],
            }
        ],
        "new_questions": [],
    }

    with pytest.raises(V5ContractError, match="unchallenged satisfied"):
        validate_summary_output(
            payload,
            {"I/O001"},
            decision_context=context,
            allowed_sources=[source],
        )
