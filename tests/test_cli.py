import pytest

from external_agents_eval.agents.config import AGENTS
from external_agents_eval.cli import build_parser


def test_deepswe_accepts_pier_native_codex_without_expanding_shared_agents():
    parser = build_parser()

    args = parser.parse_args(["deepswe", "run", "--agent", "codex"])

    assert args.agent == "codex"
    assert "codex" not in AGENTS


def test_swebench_does_not_advertise_codex_support():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["swebench", "generate", "--agent", "codex"])


def test_deepswe_accepts_composite_pi_review_repair_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "deepswe",
            "run",
            "--agent",
            "pi-review-repair",
            "--p0-manifest",
            str(tmp_path / "p0.json"),
            "--few-shot-dir",
            str(tmp_path / "few-shots"),
            "--review-arm",
            "E2",
            "--review-contract-version",
            "3.0",
            "--review-prompt-version",
            "4.0",
            "--task-name",
            "task-a",
            "--task-name",
            "task-b",
        ]
    )
    assert args.agent == "pi-review-repair"
    assert args.p0_manifest == tmp_path / "p0.json"
    assert args.review_arm == "E2"
    assert args.review_contract_version == "3.0"
    assert args.review_prompt_version == "4.0"
    assert args.task_names == ["task-a", "task-b"]


def test_deepswe_accepts_v5_review_repair_agent_without_legacy_options(tmp_path):
    args = build_parser().parse_args(
        [
            "deepswe",
            "run",
            "--agent",
            "pi-v5-review-repair",
            "--p0-manifest",
            str(tmp_path / "p0.json"),
            "--workflow-role-timeout",
            "3600",
        ]
    )

    assert args.agent == "pi-v5-review-repair"
    assert args.p0_manifest == tmp_path / "p0.json"
    assert args.workflow_role_timeout == 3600
    assert args.review_contract_version is None
    assert args.review_prompt_version is None


def test_deepswe_exposes_review_manifest_freeze_command(tmp_path):
    args = build_parser().parse_args(
        [
            "deepswe",
            "prepare-review-manifest",
            "--source-job-dir",
            str(tmp_path / "job"),
            "--output",
            str(tmp_path / "manifest.json"),
        ]
    )
    assert args.deepswe_command == "prepare-review-manifest"


def test_deepswe_exposes_experiment_specific_review_summary(tmp_path):
    args = build_parser().parse_args(
        [
            "deepswe",
            "review-summary",
            "--job-dir",
            str(tmp_path / "job"),
            "--p0-manifest",
            str(tmp_path / "p0.json"),
        ]
    )
    assert args.deepswe_command == "review-summary"
