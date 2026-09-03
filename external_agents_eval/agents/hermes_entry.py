from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PRACTICALLY_UNLIMITED_MAX_ITERATIONS = 2_147_483_647


def resolve_max_iterations(value: int | str | None) -> int:
    """Map an omitted evaluation limit to a safe integer Hermes accepts."""
    if value is None:
        return PRACTICALLY_UNLIMITED_MAX_ITERATIONS
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("max_turns must be positive when provided")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(args.input.read_text())
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    source_root = Path("/opt/hermes/source")
    sys.path.insert(0, str(source_root))
    from run_agent import AIAgent

    workdir = payload.get("workdir", "/app")
    os.chdir(workdir)
    agent = AIAgent(
        base_url=payload["base_url"],
        api_key=api_key,
        model=payload["model"],
        max_iterations=resolve_max_iterations(payload.get("max_turns")),
        enabled_toolsets=["terminal", "file"],
        save_trajectories=False,
        verbose_logging=False,
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        result = agent.run_conversation(
            payload["prompt"],
            task_id=str(payload.get("instance_id") or "external-eval-task"),
        )
        serializable = {
            "completed": result.get("completed"),
            "partial": result.get("partial", False),
            "failed": result.get("failed", False),
            "error": result.get("error"),
            "api_calls": result.get("api_calls"),
            "final_response": result.get("final_response"),
            "messages": result.get("messages", []),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        temporary.replace(args.output)
        return 0 if not serializable["failed"] else 1
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
