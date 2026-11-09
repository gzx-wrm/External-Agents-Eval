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


def _build_pi_command(
    *,
    model: str,
    prompt_path: str,
    provider: str | None = None,
    tools: tuple[str, ...] = PI_BUILTIN_TOOLS,
    session_id: str | None = None,
    session_dir: str | None = None,
    session_file: str | None = None,
    no_compaction: bool = False,
) -> str:
    cli = "/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
    tool_names = ",".join(tools)
    tool_option = "--no-tools" if not tools else f"--tools {shlex.quote(tool_names)}"
    if session_file and (session_id or session_dir):
        raise ValueError("session_file cannot be combined with session_id/session_dir")
    if session_file:
        session_option = f"--session {shlex.quote(session_file)}"
    elif session_id:
        if not session_dir:
            raise ValueError("session_dir is required with session_id")
        session_option = (
            f"--session-dir {shlex.quote(session_dir)} "
            f"--session-id {shlex.quote(session_id)}"
        )
    elif session_dir:
        raise ValueError("session_id is required with session_dir")
    else:
        session_option = "--no-session"
    # Pi 0.84.2 has no ``--no-compaction`` CLI flag.  V5 callers use the
    # option only as a signal to provision a settings file in the role
    # environment; keeping this builder free of an unknown flag is important
    # because Pi rejects unknown options before starting the session.
    del no_compaction
    command_model = model
    command_provider = provider or "deepseek"
    if provider is None and "/" in model:
        command_provider, command_model = model.split("/", 1)
    return (
        f"/opt/pi-runtime/bin/node {cli} "
        f"--mode json {session_option} --no-approve --provider {shlex.quote(command_provider)} "
        f"--model {shlex.quote(command_model)} {tool_option} "
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
        model_spec = self.model_spec_for_role("solver")
        pi_config_dir = await self.prepare_pi_config(
            environment, model_spec, config_key="solver"
        )
        command = _build_pi_command(
            model=model_spec.model,
            provider=model_spec.provider,
            prompt_path=prompt_path,
        )
        env = self.agent_env(environment)
        env.update(
            {
                "PI_CODING_AGENT_DIR": pi_config_dir,
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
            model=model_spec.model,
            agent_version=self.version(),
        )
        self.save_trajectory(trajectory, context)
        context.metadata = {
            **model_spec.to_metadata(),
            "stage_models": self.stage_model_config.to_metadata(),
            "process_exit_code": result.return_code,
        }
        if result.return_code != 0:
            output = self.redact(result.stderr or result.stdout or "no output")
            raise RuntimeError(f"Pi exited with status {result.return_code}: {output[-2000:]}")
        await self.commit_solution(environment)
