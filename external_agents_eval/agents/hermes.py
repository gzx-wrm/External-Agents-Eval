from __future__ import annotations

import json

from external_agents_eval.agents.base import AgentRunContext, AgentRunResult
from external_agents_eval.agents.config import DEFAULT_BASE_URL, get_agent_spec
from external_agents_eval.docker_io import decode_output, put_files, read_file, redact_secret


class HermesAdapter:
    spec = get_agent_spec("hermes")

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    @property
    def transcript_suffix(self) -> str:
        return ".json"

    def run(self, container, context: AgentRunContext) -> AgentRunResult:
        input_payload = {
            "instance_id": context.instance_id,
            "prompt": context.prompt,
            "model": context.model,
            "base_url": DEFAULT_BASE_URL,
            "max_turns": self.max_turns,
            "workdir": context.workdir,
        }
        put_files(
            container,
            "/",
            {"external_eval/hermes_input.json": json.dumps(input_payload)},
        )
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{context.timeout_seconds}s",
            "/opt/hermes/venv/bin/python",
            "/opt/hermes/entry.py",
            "--input",
            "/external_eval/hermes_input.json",
            "--output",
            "/external_eval/hermes_output.json",
        ]
        environment = {
            "DEEPSEEK_API_KEY": context.api_key,
            "HERMES_HOME": "/external_eval/hermes_home",
            "TERMINAL_ENV": "local",
            "TERMINAL_CWD": context.workdir,
            "HOME": "/external_eval/home",
            "PYTHONUNBUFFERED": "1",
        }
        result = container.exec_run(command, environment=environment, workdir=context.workdir)
        process_output = redact_secret(decode_output(result.output), context.api_key)
        output_payload: dict = {}
        try:
            serialized = redact_secret(
                read_file(container, "/external_eval/hermes_output.json").decode(
                    "utf-8", errors="replace"
                ),
                context.api_key,
            )
            output_payload = json.loads(serialized)
            transcript = serialized
        except Exception as error:  # noqa: BLE001 - preserve process output on any read failure
            transcript = json.dumps(
                {"process_output": process_output, "output_read_error": str(error)},
                indent=2,
            )
        if process_output:
            output_payload["process_output"] = process_output
        return AgentRunResult(
            exit_code=result.exit_code,
            transcript=transcript,
            metadata={
                "provider": "deepseek",
                "model": context.model,
                "max_turns": self.max_turns,
                "completed": output_payload.get("completed"),
                "api_calls": output_payload.get("api_calls"),
            },
        )
