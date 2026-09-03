from __future__ import annotations

import json
import shlex

from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep

from external_agents_eval.agents.config import get_agent_spec
from external_agents_eval.agents.pier.common import DeepSWEPierAgent
from external_agents_eval.agents.pier.trajectory import pi_jsonl_to_atif

PI_BUILTIN_TOOLS = ("read", "bash", "edit", "write", "grep", "find", "ls")


def _build_pi_command(*, model: str, prompt_path: str) -> str:
    cli = "/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
    tools = ",".join(PI_BUILTIN_TOOLS)
    return (
        f"/opt/pi-runtime/bin/node {cli} "
        "--mode json --no-session --no-approve --provider deepseek "
        f"--model {model} --tools {tools} "
        f"-p < {shlex.quote(prompt_path)}"
    )


class PiPierAgent(DeepSWEPierAgent):
    spec = get_agent_spec("pi")
    _NODE_VERSION = "22.19.0"

    @staticmethod
    def name() -> str:
        return "pi"

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
export NVM_DIR=/opt/pi-runtime/nvm
mkdir -p "$NVM_DIR" /opt/pi-runtime/bin
curl --retry 5 --retry-all-errors --retry-delay 2 -fsSL \
  https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
. "$NVM_DIR/nvm.sh" --no-use
nvm install {self._NODE_VERSION}
nvm use {self._NODE_VERSION}
ln -sf "$(command -v node)" /opt/pi-runtime/bin/node
ln -sf "$(command -v npm)" /opt/pi-runtime/bin/npm
npm install --prefix /opt/pi-runtime --ignore-scripts @earendil-works/pi-coding-agent@0.84.2
test "$(npm view @earendil-works/pi-coding-agent@0.84.2 version)" = "0.84.2"
""".strip()
        cli = "/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self.version(),
            cache_key=f"pi-{self.spec.upstream_commit}-node-{self._NODE_VERSION}",
            steps=[
                InstallStep(user="root", run=system_packages),
                InstallStep(user="root", run=install),
            ],
            verification_command=f"/opt/pi-runtime/bin/node {cli} --version",
            metadata={"upstream_commit": self.spec.upstream_commit},
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        settings = json.dumps({"defaultProjectTrust": "never", "telemetry": False})
        command = (
            "mkdir -p /external_eval/pi_config /external_eval/home && "
            f"printf '%s' {json.dumps(settings)} > /external_eval/pi_config/settings.json && "
            "chmod -R 777 /external_eval"
        )
        await self.checked_exec(environment, command, user="root")
        await self.checked_exec(
            environment,
            "/opt/pi-runtime/bin/node "
            "/opt/pi-runtime/node_modules/@earendil-works/"
            "pi-coding-agent/dist/cli.js --version",
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        prompt_path = await self.upload_instruction(instruction, environment)
        command = _build_pi_command(
            model=self.runtime_model,
            prompt_path=prompt_path,
        )
        env = self.agent_env(environment)
        env.update(
            {
                "PI_CODING_AGENT_DIR": "/external_eval/pi_config",
                "PI_OFFLINE": "1",
                "PI_SKIP_VERSION_CHECK": "1",
                "PI_TELEMETRY": "0",
                "HOME": "/external_eval/home",
                "PATH": (
                    "/opt/pi-runtime/bin:/usr/local/sbin:/usr/local/bin:"
                    "/usr/sbin:/usr/bin:/sbin:/bin"
                ),
            }
        )
        result = await environment.exec(command, cwd=self.workdir, env=env)
        transcript = self.redact(result.stdout)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "pi.jsonl").write_text(transcript, encoding="utf-8")
        if result.stderr:
            (self.logs_dir / "pi.stderr.txt").write_text(
                self.redact(result.stderr), encoding="utf-8"
            )
        trajectory = pi_jsonl_to_atif(
            transcript,
            instruction=instruction,
            model=self.runtime_model,
            agent_version=self.version(),
        )
        self.save_trajectory(trajectory, context)
        context.metadata = {
            "provider": "deepseek",
            "model": self.runtime_model,
            "process_exit_code": result.return_code,
        }
        if result.return_code != 0:
            output = self.redact(result.stderr or result.stdout or "no output")
            raise RuntimeError(f"Pi exited with status {result.return_code}: {output[-2000:]}")
        await self.commit_solution(environment)
