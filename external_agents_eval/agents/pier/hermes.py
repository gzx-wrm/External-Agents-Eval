from __future__ import annotations

import json
from pathlib import Path

from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep

from external_agents_eval.agents.config import DEFAULT_BASE_URL, get_agent_spec
from external_agents_eval.agents.pier.common import DeepSWEPierAgent
from external_agents_eval.agents.pier.trajectory import hermes_json_to_atif


class HermesPierAgent(DeepSWEPierAgent):
    spec = get_agent_spec("hermes")
    _UV_VERSION = "0.8.14"

    def __init__(self, *args, max_turns: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "hermes"

    def version(self) -> str:
        return self.spec.upstream_ref

    def install_spec(self) -> AgentInstallSpec:
        system_packages = """
set -eu
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bash ca-certificates curl git
  rm -rf /var/lib/apt/lists/*
elif command -v apk >/dev/null 2>&1; then
  apk add --no-cache bash ca-certificates curl git
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y bash ca-certificates curl git
elif command -v yum >/dev/null 2>&1; then
  yum install -y bash ca-certificates curl git
else
  command -v bash && command -v curl && command -v git
fi
""".strip()
        install = f"""
set -eu
curl -LsSf https://astral.sh/uv/{self._UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
git clone --depth 1 --branch {self.spec.upstream_ref} {self.spec.upstream_url}.git /opt/hermes/source
test "$(git -C /opt/hermes/source rev-parse HEAD)" = "{self.spec.upstream_commit}"
UV_PYTHON_INSTALL_DIR=/opt/hermes/python uv python install 3.11
python_bin="$(find /opt/hermes/python -type f -path '*/bin/python3.11' | head -n 1)"
test -n "$python_bin"
uv venv --python "$python_bin" --relocatable /opt/hermes/venv
cd /opt/hermes/source
UV_PROJECT_ENVIRONMENT=/opt/hermes/venv uv sync --locked --no-dev
""".strip()
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self.version(),
            cache_key=f"hermes-{self.spec.upstream_commit}-uv-{self._UV_VERSION}",
            steps=[
                InstallStep(user="root", run=system_packages),
                InstallStep(user="root", run=install),
            ],
            verification_command=(
                "/opt/hermes/venv/bin/python -c "
                "\"import sys; sys.path.insert(0, '/opt/hermes/source'); "
                "from run_agent import AIAgent; print('Hermes import OK')\""
            ),
            metadata={"upstream_commit": self.spec.upstream_commit},
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        await self.checked_exec(environment, "mkdir -p /opt/hermes", user="root")
        entry = Path(__file__).resolve().parents[1] / "hermes_entry.py"
        await environment.upload_file(entry, "/opt/hermes/entry.py")
        await self.checked_exec(
            environment,
            "/opt/hermes/venv/bin/python -m py_compile /opt/hermes/entry.py",
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        input_payload = {
            "instance_id": self.logs_dir.parent.name,
            "prompt": instruction,
            "model": self.runtime_model,
            "base_url": DEFAULT_BASE_URL,
            "max_turns": self.max_turns,
            "workdir": self.workdir,
        }
        host_input = self.logs_dir / "hermes-input.json"
        self.save_json(host_input, input_payload)
        environment_input = "/tmp/external-agent-hermes-input.json"
        environment_output = "/tmp/external-agent-hermes-output.json"
        await environment.upload_file(host_input, environment_input)
        command = (
            "/opt/hermes/venv/bin/python /opt/hermes/entry.py "
            f"--input {environment_input} --output {environment_output}"
        )
        env = self.agent_env(environment)
        env.update(
            {
                "HERMES_HOME": "/external_eval/hermes_home",
                "TERMINAL_ENV": "local",
                "TERMINAL_CWD": self.workdir,
                "HOME": "/external_eval/home",
                "PYTHONUNBUFFERED": "1",
            }
        )
        await self.checked_exec(
            environment,
            "mkdir -p /external_eval/hermes_home /external_eval/home && "
            "chmod -R 777 /external_eval",
            user="root",
        )
        result = await environment.exec(command, cwd=self.workdir, env=env)
        if result.stdout:
            (self.logs_dir / "hermes.stdout.txt").write_text(
                self.redact(result.stdout), encoding="utf-8"
            )
        if result.stderr:
            (self.logs_dir / "hermes.stderr.txt").write_text(
                self.redact(result.stderr), encoding="utf-8"
            )
        host_output = self.logs_dir / "hermes-output.json"
        try:
            await environment.download_file(environment_output, host_output)
            serialized = self.redact(host_output.read_text(encoding="utf-8"))
            output_payload = self.redact_payload(json.loads(serialized))
            self.save_json(host_output, output_payload)
        except Exception as error:  # noqa: BLE001 - environment backends vary
            output_payload = {
                "completed": False,
                "failed": True,
                "error": f"Cannot retrieve Hermes output: {error}",
                "messages": [],
            }
            self.save_json(host_output, output_payload)
        trajectory = hermes_json_to_atif(
            output_payload,
            instruction=instruction,
            model=self.runtime_model,
            agent_version=self.version(),
        )
        self.save_trajectory(trajectory, context)
        context.metadata = {
            "provider": "deepseek",
            "model": self.runtime_model,
            "max_turns": self.max_turns,
            "api_calls": output_payload.get("api_calls"),
            "process_exit_code": result.return_code,
        }
        if result.return_code != 0:
            output = self.redact(
                result.stderr or result.stdout or output_payload.get("error") or "no output"
            )
            raise RuntimeError(
                f"Hermes exited with status {result.return_code}: {str(output)[-2000:]}"
            )
        await self.commit_solution(environment)
