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
