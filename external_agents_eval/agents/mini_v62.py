"""Mini-SWE-agent backend for the existing V6.2 controller and review protocol."""
from __future__ import annotations

import asyncio
import json
import posixpath
import shlex
import tarfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from external_agents_eval.agents.pi_file_tools import (
    PI_EVIDENCE_PROTOCOL,
    PI_TOOL_NAMES,
    execute_tool,
    load_tool_schemas,
    prepare_tools,
)
from external_agents_eval.agents.pier.pi_role_runner import (
    ContainerGitWorkspaceManager,
    PiRoleRunner,
)
from external_agents_eval.agents.pier.pi_v62_solve_review_repair import (
    PiV62SolveReviewRepairPierAgent,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes, sha256_json
from external_agents_eval.workflows.solve_review_repair.bash_evidence import PROTOCOL
from external_agents_eval.workflows.solve_review_repair.v5_config import V5_ROLE_STEP_LIMIT
from external_agents_eval.workflows.solve_review_repair.models import (
    IndeterminateInvocationError,
    RoleName,
    RoleRunResult,
)

MINI_SYSTEM = """You are an assistant operating through the bash tool.
Your assigned working directory is {{cwd}}. Edit and run tests only in that
working tree. Other paths supplied in the stage input are read-only evidence.
/testbed is the original baseline; do not edit it or run candidate tests there.
Follow the supplied stage assignment. Execute one bash command per response.
When finished, run a command whose first output line is exactly
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, followed by your final answer (the complete
JSON object when requested). Do not submit a patch in place of a requested JSON
report. Source changes remain in the working tree. Do not commit them.
You can submit a JSON answer directly with: submit_json '<complete JSON>'
The sandbox is offline. Do not access remote issue discussions or solutions.
Use normal shell commands to inspect files: cat, sed -n, head, tail or nl.
During A exploration, the harness records source text actually returned by Bash;
directory listings and grep/find/rg output are discovery information. Read the
relevant source and documentation before finishing. You do not need to select or
resubmit evidence: collection is automatic and does not use your final draft.
"""

PI_FILE_SYSTEM = """You are an expert coding assistant inspecting a repository.
Your assigned working directory is {{cwd}}. Follow the supplied stage assignment.
Available tools are read, grep, find, and ls. Use read to examine file contents;
use grep to search contents and find or ls to discover files. Read relevant source
and documentation before finishing. Use offset/limit to continue truncated reads.
When finished, respond with the final answer requested by the stage assignment.
"""


def parse_pi_actions(response, schemas):
    """Accept native file-tool calls, or Pi-style completion without a tool call."""
    from jsonschema import Draft202012Validator
    from minisweagent.exceptions import FormatError

    message = response.choices[0].message
    calls = message.tool_calls or []
    if not calls and message.content and message.content.strip():
        return [{"final": message.content}]
    definitions = {item["function"]["name"]: item["function"]["parameters"] for item in schemas}
    try:
        if not calls:
            raise ValueError("Call a file tool or provide your final response.")
        actions = []
        for call in calls:
            name = call.function.name
            if name not in definitions:
                raise ValueError(f"Unknown tool '{name}'. Available tools: {', '.join(definitions)}")
            arguments = json.loads(call.function.arguments)
            errors = list(Draft202012Validator(definitions[name]).iter_errors(arguments))
            if errors:
                raise ValueError(errors[0].message)
            actions.append({"name": name, "arguments": arguments, "tool_call_id": call.id})
        return actions
    except ValueError as error:
        raise FormatError({"role": "user", "content": str(error),
                           "extra": {"interrupt_type": "FormatError"}}) from error


class MiniPiFileEnvironment:
    """Only registered Pi file tools execute; arbitrary Bash has no dispatch path."""

    def __init__(self, bridge, cwd):
        self.bridge = bridge
        self.cwd = cwd
        self.events = []

    def get_template_vars(self, **kwargs):
        return {"cwd": self.cwd}

    def serialize(self):
        return {"info": {"environment": {"cwd": self.cwd, "network": "none",
                                          "tools": list(PI_TOOL_NAMES)}}}

    def execute(self, action):
        from minisweagent.exceptions import Submitted

        if "final" in action:
            raise Submitted({"role": "exit", "content": action["final"],
                             "extra": {"exit_status": "Submitted", "submission": action["final"]}})
        name = action["name"]
        if name not in PI_TOOL_NAMES:
            raise ValueError(f"Unknown Pi file tool '{name}'")
        arguments = dict(action["arguments"])
        result = execute_tool(self.bridge, self.cwd, name, arguments, action["tool_call_id"])
        if result.get("resolved_path"):
            arguments["path"] = result["resolved_path"]
        arguments["mini_tool_call_id"] = action["tool_call_id"]
        self.events.append((name, arguments, result))
        return result


def helper_call(command, cwd):
    """Accept a standalone helper, including the usual cd/bash shell wrappers."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    words = list(lexer)
    if len(words) == 3 and words[0] in {"bash", "/bin/bash"} and words[1] in {"-c", "-lc"}:
        nested = shlex.shlex(words[2], posix=True, punctuation_chars=";&|<>()")
        nested.whitespace_split = True
        words = list(nested)
    while words and words[-1] == ";":
        words.pop()
    if len(words) == 5 and words[0] == "cd" and words[2] == "&&":
        cwd = posixpath.normpath(posixpath.join(cwd, words[1]))
        words = words[3:]
    if len(words) == 2 and words[0] in {"ft_read", "submit_json"}:
        return words[0], words[1], cwd
    return None


def close_tool_calls(messages, submission):
    """A submission interrupts mini before it appends the final tool response."""
    messages = [message for message in messages if message["role"] != "exit"]
    answered = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
    for message in list(messages):
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                if call["id"] not in answered:
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + submission})
    return messages


def model_visible_events(events, messages):
    """Mini can interrupt a batch before its executed tool results are appended."""
    delivered = {message["tool_call_id"]: message for message in messages
                 if message.get("role") == "tool" and "tool_call_id" in message}
    visible, unreturned = [], []
    for name, arguments, result in events:
        call_id = arguments.get("mini_tool_call_id")
        if call_id and call_id not in delivered:
            unreturned.append({"tool_call_id": call_id, "command": arguments.get("command"),
                               "output_sha256": sha256_bytes(result["output"].encode())})
            continue
        if call_id and result["output"] not in str(delivered[call_id].get("content", "")):
            raise ValueError("Mini observation changed or truncated the raw output; refuse unseen evidence")
        visible.append((name, arguments, result))
    return visible, unreturned


class MiniBashEnvironment:
    def __init__(self, bridge, cwd, tools=None):
        self.bridge = bridge
        self.cwd = cwd
        self.tools = tools
        self.events = []

    def get_template_vars(self, **kwargs):
        return {"cwd": self.cwd}

    def serialize(self):
        return {"info": {"environment": {"cwd": self.cwd, "network": "none"}}}

    def _command_output(self, raw):
        """What a Bash command returns to the model (subclasses may bound it)."""
        return raw.stdout + raw.stderr

    def execute(self, action):
        from minisweagent.exceptions import Submitted

        command = action["command"]
        try:
            helper = helper_call(command, self.cwd)
        except ValueError:
            helper = None
        name, arguments = "bash", {"command": command}
        if helper and helper[0] == "submit_json":
            result = {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + helper[1],
                      "returncode": 0, "exception_info": ""}
        elif self.tools == ():
            result = {"output": "Filesystem tools are disabled. Use submit_json '<complete JSON>'.",
                      "returncode": 126, "exception_info": ""}
        elif helper and helper[0] == "ft_read":
            name = "read"
            try:
                arguments = json.loads(helper[1])
                path = arguments["path"]
                if not path.startswith("/"):
                    path = posixpath.normpath(posixpath.join(helper[2], path))
                arguments["path"] = path
                raw = self.bridge.exec_sync("cat -- " + shlex.quote(path), cwd=self.cwd)
                if raw.return_code:
                    raise ValueError(raw.stderr)
                lines = raw.stdout.splitlines()
                start = int(arguments.get("offset", 1))
                limit = int(arguments.get("limit", 2000))
                if start < 1 or limit < 1:
                    raise ValueError("offset and limit must be positive")
                selected = []
                size = 0
                for line in lines[start - 1:start - 1 + min(limit, 2000)]:
                    needed = len((line + "\n").encode())
                    if size + needed > 50 * 1024:
                        break
                    selected.append(line)
                    size += needed
                output = "\n".join(selected)
                end = start - 1 + len(selected)
                if end < len(lines):
                    if not selected:
                        raise ValueError("A source line exceeds the read byte budget; choose another range")
                    output += f"\n\n[{len(lines) - end} more lines in file. Use offset={end + 1} to continue.]"
                result = {"output": output, "returncode": 0, "exception_info": ""}
            except (ValueError, KeyError, IndexError) as error:
                name, arguments = "bash", {"command": command}
                result = {"output": str(error), "returncode": 1, "exception_info": "ft_read failed"}
        else:
            raw = self.bridge.exec_sync(command, cwd=self.cwd)
            arguments["cwd"] = self.cwd
            result = {"output": self._command_output(raw), "returncode": raw.return_code, "exception_info": ""}
        if action.get("tool_call_id"):
            arguments["mini_tool_call_id"] = action["tool_call_id"]
        self.events.append((name, arguments, result))
        lines = result["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and result["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission,
                             "extra": {"exit_status": "Submitted", "submission": submission}})
        return result


class MiniRoleRunner(PiRoleRunner):
    """Reuse artifact transport helpers, with every agent loop executed by mini."""

    def _make_mini_environment(self, *, role, prompt, worktree, tools, schemas, host):
        return MiniPiFileEnvironment(self.environment, worktree) if schemas else MiniBashEnvironment(
            self.environment, worktree, tools=tools,
        )

    def _mini_system(self, *, role, tools, schemas):
        system = PI_FILE_SYSTEM if schemas else MINI_SYSTEM
        if not schemas and tools and "bash" not in tools:
            system += "\nThis stage is read-only: do not edit source or run new tests. Use normal Bash reads.\n"
        if not schemas and not tools:
            system += "\nFor this formatting stage, the filesystem and shell are disabled. Submit using only: submit_json '<complete JSON>'\n"
        return system

    def _make_mini_agent(self, *, model, env, system, timeout_sec):
        from minisweagent.agents.default import DefaultAgent

        return DefaultAgent(
            model, env, system_template=system, instance_template="{{task}}",
            step_limit=V5_ROLE_STEP_LIMIT, cost_limit=0, wall_time_limit_seconds=int(timeout_sec or 1800),
        )

    async def run_role(self, *, role, prompt, worktree, tools, timeout_sec,
                       artifact_prefix, session_id=None, session_dir=None,
                       session_file=None, no_compaction=False, invocation_id=None):
        pi_runtime = await prepare_tools(self.environment) if role == RoleName.V5_SPECIFICATION_REVIEWER else None
        result = await asyncio.to_thread(
            self._run_mini, role, prompt, worktree, timeout_sec, artifact_prefix, session_id, tools, pi_runtime,
        )
        if result.session_file is not None:
            await self.environment.upload_file(result.session_file, result.remote_session_file)
        return result

    def _run_mini(self, role, prompt, worktree, timeout_sec, artifact_prefix, session_id, tools, pi_runtime=None):
        from minisweagent.models.litellm_model import LitellmModel

        spec = self.stage_model_config.for_role(role)
        host = self.logs_root / artifact_prefix
        host.mkdir(parents=True, exist_ok=True)
        identity = {"prompt_sha256": sha256_bytes(prompt.encode()), "model": spec.to_metadata(),
                    "worktree": worktree, "session_id": session_id}
        schemas = load_tool_schemas(pi_runtime) if pi_runtime else None
        if schemas:
            runtime_metadata = json.loads((pi_runtime / "runtime.json").read_text())
            identity.update(tool_protocol=PI_EVIDENCE_PROTOCOL, tool_schemas_sha256=sha256_json(schemas),
                            tool_runtime_sha256=runtime_metadata["archive_sha256"],
                            tool_helper_sha256=sha256_bytes(Path(__file__).with_name("pi_file_tools.mjs").read_bytes()))
        request = host / "mini-invocation.json"
        completed = host / "role-result.json"
        if request.exists():
            if json.loads(request.read_text()) != identity:
                raise ValueError("Mini invocation inputs changed; use a new run")
            if not completed.exists():
                raise IndeterminateInvocationError("Mini invocation has no completed result; do not resample")
            saved = json.loads(completed.read_text())
            for key in ("transcript_path", "trajectory_path", "session_file"):
                if saved.get(key):
                    saved[key] = Path(saved[key])
            return RoleRunResult(**saved)
        request.write_text(json.dumps(identity))
        if schemas:
            (host / "tool-schemas.json").write_text(json.dumps(schemas, indent=2))
            (host / "tool-runtime.json").write_text(json.dumps(runtime_metadata, indent=2))
        (host / "prompt.md").write_text(prompt)
        key = sha256_json([session_id or artifact_prefix, spec.to_metadata()])
        session = self.logs_root / "mini-sessions" / f"{key}.json"
        session.parent.mkdir(exist_ok=True)
        prefix = json.loads(session.read_text()) if session.exists() else []
        pre_hash = sha256_bytes(session.read_bytes()) if session.exists() else None
        progress = {"model_calls": 0, "completed_calls": 0}

        def save_progress(**changes):
            progress.update(changes)
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            temporary = host / "progress.json.tmp"
            temporary.write_text(json.dumps(progress))
            temporary.replace(host / "progress.json")

        class Model(LitellmModel):
            def _query(self, messages, **kwargs):
                if schemas is None:
                    return super()._query(messages, **kwargs)
                import litellm

                options = self.config.model_kwargs | kwargs
                options["tools"] = schemas
                return litellm.completion(model=self.config.model_name, messages=messages, **options)

            def _parse_actions(self, response):
                if schemas is None:
                    return super()._parse_actions(response)
                return parse_pi_actions(response, schemas)

            def format_observation_messages(self, message, outputs, template_vars=None):
                if schemas is None:
                    return super().format_observation_messages(message, outputs, template_vars)
                from minisweagent.models.utils.actions_toolcall import (
                    format_toolcall_observation_messages,
                )

                return format_toolcall_observation_messages(
                    actions=message["extra"]["actions"], outputs=outputs,
                    observation_template="{{output.output}}", template_vars=template_vars,
                )

            def _prepare_messages_for_api(self, messages):
                prepared = super()._prepare_messages_for_api(messages)
                if spec.provider == "deepseek":
                    for message in prepared:
                        if message.get("role") == "assistant" and message.get("reasoning_content") == "":
                            message["reasoning_content"] = " "
                return prepared

            def query(self, messages, **kwargs):
                save_progress(model_calls=progress["model_calls"] + 1, state="awaiting_response")
                try:
                    response = super().query(prefix + messages[1:] if prefix else messages, **kwargs)
                except Exception as exception:
                    save_progress(state="query_error", error_type=type(exception).__name__)
                    raise
                usage = (response.get("extra", {}).get("response") or {}).get("usage") or {}
                save_progress(completed_calls=progress["completed_calls"] + 1,
                              state="response_received", last_usage=usage)
                return response

        model = Model(
            model_name=f"{spec.provider}/{spec.model}", cost_tracking="ignore_errors",
            model_kwargs={"api_base": spec.effective_base_url, "reasoning_effort": "high",
                          "extra_body": {"thinking": {"type": "enabled"}}, "max_tokens": 65536,
                          "timeout": 180, "num_retries": 0},
        )
        env = self._make_mini_environment(
            role=role, prompt=prompt, worktree=worktree, tools=tools, schemas=schemas, host=host,
        )
        system = self._mini_system(role=role, tools=tools, schemas=schemas)
        agent = self._make_mini_agent(model=model, env=env, system=system, timeout_sec=timeout_sec)
        started = datetime.now(timezone.utc)
        status, error = {}, None
        try:
            status = agent.run(prompt)
        except Exception as exception:  # noqa: BLE001 - archive the failed agent invocation
            error = self.redact(str(exception))
        raw = agent.serialize()
        (host / "mini.json").write_text(self.redact(json.dumps(raw, ensure_ascii=False, indent=2)))
        submission = status.get("submission", "")
        current = close_tool_calls(agent.messages, submission)
        full = prefix + current[1:] if prefix else current
        session.write_text(self.redact(json.dumps(full, ensure_ascii=False)))
        totals = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_cached_tokens": 0}
        for message in agent.messages:
            usage = (message.get("extra", {}).get("response") or {}).get("usage") or {}
            totals["total_prompt_tokens"] += usage.get("prompt_tokens") or 0
            totals["total_completion_tokens"] += usage.get("completion_tokens") or 0
            totals["total_cached_tokens"] += usage.get("prompt_cache_hit_tokens") or (
                usage.get("prompt_tokens_details") or {}
            ).get("cached_tokens", 0)
        steps = [{"step_id": 1, "source": "user", "message": prompt}]
        visible, unreturned = model_visible_events(env.events, agent.messages)
        for name, arguments, output in visible:
            call_id = f"mini-{len(steps)}"
            steps.append({"step_id": len(steps) + 1, "source": "agent", "message": "",
                          "tool_calls": [{"tool_call_id": call_id, "function_name": name, "arguments": arguments}],
                          "observation": {"results": [{"source_call_id": call_id, "content": output["output"],
                                                       "exit_code": output["returncode"],
                                                       "source_text_returned": output.get("source_text_returned", True)}]}})
        steps.append({"step_id": len(steps) + 1, "source": "agent", "message": submission})
        trajectory = host / "trajectory.json"
        trajectory.write_text(self.redact(json.dumps({
            "schema_version": "ATIF-v1.7", "agent": {"name": "mini-swe-agent", "version": "2.4.6",
                                                       "model_name": spec.model},
            "steps": steps, "final_metrics": {"total_steps": len(steps), **totals},
            "evidence_protocol": PI_EVIDENCE_PROTOCOL if schemas else PROTOCOL,
            "executed_but_not_returned": unreturned,
            "adapter": "Pi 0.84.2 file tools; raw mini.json is authoritative" if schemas else
                       "native Bash observations; raw mini.json is authoritative",
        }, ensure_ascii=False)))
        result = RoleRunResult(
            exit_code=0 if status.get("exit_status") == "Submitted" else 1,
            final_assistant_text=submission, transcript_path=host / "mini.json", trajectory_path=trajectory,
            started_at=started.isoformat(), finished_at=datetime.now(timezone.utc).isoformat(),
            resource_usage=totals, timed_out=status.get("exit_status") == "TimeExceeded",
            exception=error or (None if status.get("exit_status") == "Submitted" else str(status)),
            session_file=session, pre_session_sha256=pre_hash,
            post_session_sha256=sha256_bytes(session.read_bytes()),
            remote_session_file=f"/external_eval/mini_sessions/{key}.json",
        )
        completed.write_text(self.redact(json.dumps(asdict(result), default=str)))
        return result


class MiniWorkspaceManager(ContainerGitWorkspaceManager):
    """Give each worktree the baseline image's ignored compiled runtime assets."""

    async def read_frozen_text_files(self, *, worktree, base_commit):
        """Read the frozen tree once; the evidence packet still includes only visible lines."""
        token = uuid.uuid4().hex
        remote = f"{self.INPUT_ROOT}/bash-baseline-{token}.tar"
        local = self.transfer_dir / f"bash-baseline-{token}.tar"
        self.transfer_dir.mkdir(parents=True, exist_ok=True)
        try:
            # git archive honors export-ignore/export-subst; read blob objects instead.
            script = """import io, subprocess, sys, tarfile
tree = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', sys.argv[1]])
process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
with tarfile.open(sys.argv[2], 'w', format=tarfile.PAX_FORMAT, encoding='utf-8') as archive:
    for entry in tree.split(b'\\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\\t', 1)
        mode, kind, oid = metadata.split()
        if kind != b'blob':
            continue
        process.stdin.write(oid + b'\\n')
        process.stdin.flush()
        header = process.stdout.readline().split()
        raw = process.stdout.read(int(header[2]))
        assert process.stdout.read(1) == b'\\n'
        member = tarfile.TarInfo(name.decode('utf-8', 'surrogateescape'))
        if mode == b'120000':
            member.type = tarfile.SYMTYPE
            member.linkname = raw.decode('utf-8', 'surrogateescape')
            archive.addfile(member)
        else:
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
process.stdin.close()
process.stdout.close()
assert process.wait() == 0
"""
            await self._checked(
                "python -c " + shlex.quote(script) + " " + shlex.quote(base_commit)
                + " " + shlex.quote(remote),
                cwd=worktree,
            )
            await self.environment.download_file(remote, local)
            files, links = {}, {}
            with tarfile.open(local) as archive:
                for member in archive:
                    if member.issym():
                        links[member.name] = posixpath.normpath(posixpath.join(
                            posixpath.dirname(member.name), member.linkname,
                        ))
                    elif member.isfile():
                        raw = archive.extractfile(member).read()
                        if b"\0" in raw:
                            continue
                        try:
                            raw.decode()
                        except UnicodeDecodeError:
                            continue
                        files[member.name] = raw
            # Match the historical FT rule: resolve one symlink hop only.
            for path, target in links.items():
                if target in files and target not in links:
                    files[path] = files[target]
            return files
        finally:
            local.unlink(missing_ok=True)
            await self.environment.exec("rm -f -- " + shlex.quote(remote))

    async def prepare(self, *, base_commit):
        paths = await super().prepare(base_commit=base_commit)
        helper = Path(__file__).with_name("mini_runtime_assets.py")
        self._runtime_helper = "/external_eval/mini_runtime_assets.py"
        self._runtime_snapshot = "/external_eval/mini_runtime_assets"
        await self.environment.upload_file(helper, self._runtime_helper)
        result = await self._checked(
            f"python {self._runtime_helper} snapshot --source {shlex.quote(self.repository)} "
            f"--snapshot {self._runtime_snapshot}",
        )
        (self.transfer_dir.parent / "runtime-assets.json").write_text(result.stdout)
        for worktree in (paths.solver, paths.spec_review, paths.patch_review, paths.repair, paths.summary):
            if worktree:
                await self._seed_runtime(worktree)
        return paths

    async def _seed_runtime(self, worktree):
        await self._checked(
            f"python {self._runtime_helper} seed --snapshot {self._runtime_snapshot} "
            f"--worktree {shlex.quote(worktree)}",
        )

    async def prepare_additional_worktree(self, **kwargs):
        worktree = await super().prepare_additional_worktree(**kwargs)
        await self._seed_runtime(worktree)
        return worktree

    async def restore_reviewer_worktree(self, **kwargs):
        discarded = await super().restore_reviewer_worktree(**kwargs)
        await self._seed_runtime(kwargs["worktree"])
        return discarded


class MiniV62Agent(PiV62SolveReviewRepairPierAgent):
    workdir = "/testbed"

    def version(self):
        return "2.4.6"

    @staticmethod
    def name():
        return "mini-v62-review-repair"

    def _make_role_runner(self, **kwargs):
        return MiniRoleRunner(**kwargs)

    def _make_workspace(self, **kwargs):
        return MiniWorkspaceManager(**kwargs)

    async def commit_solution(self, environment):
        # The host submits outputs/v5-p1.patch to the official verifier.
        return None

    def save_trajectory(self, payload, context):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(json.dumps(payload, ensure_ascii=False))
