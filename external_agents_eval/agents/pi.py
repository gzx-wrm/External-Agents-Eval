from __future__ import annotations

import json

from external_agents_eval.agents.base import AgentRunContext, AgentRunResult
from external_agents_eval.agents.config import get_agent_spec
from external_agents_eval.docker_io import decode_output, put_files, redact_secret


class PiAdapter:
    spec = get_agent_spec("pi")

    @property
    def transcript_suffix(self) -> str:
        return ".jsonl"

    def run(self, container, context: AgentRunContext) -> AgentRunResult:
        config = {
            "defaultProjectTrust": "never",
            "telemetry": False,
        }
        put_files(
            container,
            "/",
            {
                "external_eval/pi_config/settings.json": json.dumps(config),
            },
        )
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{context.timeout_seconds}s",
            "/opt/pi-runtime/bin/node",
            "/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js",
            "--mode",
            "json",
            "--no-session",
            "--no-approve",
            "--provider",
            "deepseek",
            "--model",
            context.model,
            "--tools",
            "read,write,edit,bash",
            "-p",
            context.prompt,
        ]
        environment = {
            "DEEPSEEK_API_KEY": context.api_key,
            "PI_CODING_AGENT_DIR": "/external_eval/pi_config",
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            "HOME": "/external_eval/home",
            "PATH": "/opt/pi-runtime/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
        result = container.exec_run(command, environment=environment, workdir=context.workdir)
        transcript = redact_secret(decode_output(result.output), context.api_key)
        return AgentRunResult(
            exit_code=result.exit_code,
            transcript=transcript,
            metadata={
                "mode": "json",
                "provider": "deepseek",
                "model": context.model,
            },
        )
