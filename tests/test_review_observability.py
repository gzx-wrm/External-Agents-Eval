import json

from external_agents_eval.workflows.solve_review_repair.observability import (
    observe_repair_sequence,
)


def write_trajectory(path, calls):
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "step_id": index + 1,
                        "tool_calls": [call],
                    }
                    for index, call in enumerate(calls)
                ]
            }
        )
    )


def test_repair_sequence_observes_schema_read_before_explicit_edit(tmp_path):
    trajectory = tmp_path / "trajectory.json"
    write_trajectory(
        trajectory,
        [
            {
                "function_name": "read",
                "arguments": {"path": "/contracts/repair_audit.schema.json"},
            },
            {
                "function_name": "read",
                "arguments": {"path": "/contracts/source_policy.json"},
            },
            {
                "function_name": "edit",
                "arguments": {"path": "/worktree/client.py"},
            },
        ],
    )

    result = observe_repair_sequence(
        trajectory,
        audit_schema_path="/contracts/repair_audit.schema.json",
        audit_output_path="/contracts/repair_audit.json",
        frozen_inputs={"source_policy": "/contracts/source_policy.json"},
    )

    assert result["status"] == "schema_read_before_explicit_repository_write"
    assert result["audit_schema_read_before_explicit_write"] is True
    assert result["frozen_input_reads_before_explicit_write"] == ["source_policy"]
    assert result["first_explicit_repository_write"]["step_id"] == 3


def test_repair_sequence_does_not_treat_audit_output_write_as_repository_edit(tmp_path):
    trajectory = tmp_path / "trajectory.json"
    write_trajectory(
        trajectory,
        [
            {
                "function_name": "write",
                "arguments": {"path": "/contracts/repair_audit.json"},
            }
        ],
    )

    result = observe_repair_sequence(
        trajectory,
        audit_schema_path="/contracts/repair_audit.schema.json",
        audit_output_path="/contracts/repair_audit.json",
        frozen_inputs={},
    )

    assert result["status"] == "no_explicit_repository_write_observed"
    assert result["first_explicit_repository_write"] is None
