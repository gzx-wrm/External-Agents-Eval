"""V6.3: native mini Bash file delivery over the unchanged V6.2 controller."""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from external_agents_eval.agents.mini_review_files import (
    PROTOCOL,
    READ_RECORDS,
    adapt_prompt,
    collect_batch,
    controller_state,
)
from external_agents_eval.agents.mini_v62 import MiniRoleRunner, MiniV62Agent
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v5_contracts import V5ContractError
from external_agents_eval.workflows.solve_review_repair.v63_config import V63Config
from external_agents_eval.workflows.solve_review_repair.v5_config import V5_ROLE_STEP_LIMIT

REVIEW_ROLES = {
    RoleName.V5_IMPLEMENTATION_REVIEWER: "I",
    RoleName.V5_REGRESSION_REVIEWER: "R",
}

REVIEW_SYSTEM = """You are a coding reviewer operating through the native Bash tool.
Your assigned working directory is {{cwd}}. Follow your role and the authoritative
V6 controller state. Do not edit tracked source or commit changes. Use the current
$REVIEW_OUTPUT_ROOT for review records, probes and logs. The sandbox is offline;
do not access remote solutions, hidden tests, grading outputs or other reviewers.
Execute one Bash action per response. Deliver your review through Markdown files
as specified in V6.3 FILE DELIVERY, not through a final answer. No submit_json or
finish tool is needed. The adapter validates files after each command, returns
actionable errors, and hands a valid closed batch to the V6 controller automatically.
"""


class MiniFileReviewEnvironment:
    """Execute normal Bash; inspect files without ending mini inside execute()."""

    def __init__(self, bridge, cwd, *, role, state, archive, redact=lambda text: text):
        self.bridge, self.cwd, self.role = bridge, cwd, role
        self.state, self.archive, self.redact = state, Path(archive), redact
        self.output_root = state["review_notes"]["output_root"]
        self.events = []
        self.batch = None
        self.record_hashes = {}

    def get_template_vars(self, **kwargs):
        return {"cwd": self.cwd}

    def serialize(self):
        return {"info": {"environment": {
            "cwd": self.cwd, "network": "none", "review_delivery": PROTOCOL,
            "review_output_root": self.output_root,
        }}}

    def _command_output(self, raw):
        """What a Bash command returns to the model (subclasses may bound it)."""
        return raw.stdout + raw.stderr

    def execute(self, action):
        command = action["command"]
        # This is an ordinary shell environment variable, not a pseudo-command.
        raw = self.bridge.exec_sync(
            "export REVIEW_OUTPUT_ROOT=" + shlex.quote(self.output_root) + "\n" + command,
            cwd=self.cwd,
        )
        self.batch = None
        self.record_hashes = {}
        scan = self.bridge.exec_sync(
            "python -c " + shlex.quote(READ_RECORDS) + " " + shlex.quote(self.output_root),
            cwd=self.cwd,
        )
        error = None
        try:
            if scan.return_code:
                raise V5ContractError("Cannot inspect current review records: " + scan.stderr[-1000:])
            snapshot = json.loads(scan.stdout)
            if "error" in snapshot:
                raise V5ContractError(snapshot["error"])
            files = snapshot["files"]
            self.record_hashes = {
                path: sha256_bytes(text.encode("utf-8")) for path, text in files.items()
            }
            self.batch, message = collect_batch(
                files, role=self.role, state=self.state, archive=self.archive,
            )
        except (V5ContractError, ValueError, KeyError) as failure:
            error = str(failure)
            message = "Fix current files before continuing: " + error
        feedback = f"\n[V6.3 review files; root={self.output_root}] {message}\n"
        result = {
            "output": self._command_output(raw) + feedback,
            "returncode": raw.return_code, "exception_info": "",
        }
        arguments = {"command": command, "cwd": self.cwd}
        if action.get("tool_call_id"):
            arguments["mini_tool_call_id"] = action["tool_call_id"]
        self.events.append(("bash", arguments, result))
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "protocol": PROTOCOL, "output_root": self.output_root,
            "record_sha256": self.record_hashes, "error": error,
            "batch": self.batch,
        }
        (self.archive.parent / "file-review.json").write_text(
            self.redact(json.dumps(receipt, ensure_ascii=False, indent=2)), encoding="utf-8",
        )
        return result


class MiniV63RoleRunner(MiniRoleRunner):
    async def run_role(self, *, role, prompt, worktree, tools, timeout_sec,
                       artifact_prefix, session_id=None, session_dir=None,
                       session_file=None, no_compaction=False, invocation_id=None):
        if role in REVIEW_ROLES:
            adapted = self._adapt_review_prompt(prompt)
            host = self.logs_root / artifact_prefix
            host.mkdir(parents=True, exist_ok=True)
            original = host / "controller-prompt.md"
            if original.exists() and original.read_text() != self.redact(prompt):
                raise ValueError("Original Reviewer controller input changed; use a new invocation")
            original.write_text(self.redact(prompt))
            prompt = adapted
        return await super().run_role(
            role=role, prompt=prompt, worktree=worktree, tools=tools, timeout_sec=timeout_sec,
            artifact_prefix=artifact_prefix, session_id=session_id, session_dir=session_dir,
            session_file=session_file, no_compaction=no_compaction, invocation_id=invocation_id,
        )

    def _adapt_review_prompt(self, prompt):
        return adapt_prompt(prompt)

    def _make_mini_environment(self, *, role, prompt, worktree, tools, schemas, host):
        if role in REVIEW_ROLES:
            return MiniFileReviewEnvironment(
                self.environment, worktree, role=REVIEW_ROLES[role],
                state=controller_state(prompt), archive=host / "review-records", redact=self.redact,
            )
        return super()._make_mini_environment(
            role=role, prompt=prompt, worktree=worktree, tools=tools, schemas=schemas, host=host,
        )

    def _mini_system(self, *, role, tools, schemas):
        if role in REVIEW_ROLES:
            return REVIEW_SYSTEM
        return super()._mini_system(role=role, tools=tools, schemas=schemas)

    def _make_mini_agent(self, *, model, env, system, timeout_sec):
        if not isinstance(env, MiniFileReviewEnvironment):
            return super()._make_mini_agent(
                model=model, env=env, system=system, timeout_sec=timeout_sec,
            )
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.exceptions import Submitted

        class FileReviewAgent(DefaultAgent):
            def execute_actions(self, message):
                # Add the REAL command result before ending the batch. Otherwise
                # mini's Submitted exception would lose the final tool observation.
                observations = super().execute_actions(message)
                if env.batch is not None:
                    submission = json.dumps(env.batch, ensure_ascii=False)
                    raise Submitted({
                        "role": "exit", "content": submission,
                        "extra": {"exit_status": "Submitted", "submission": submission,
                                  "review_delivery": PROTOCOL,
                                  "record_sha256": env.record_hashes},
                    })
                return observations

        return FileReviewAgent(
            model, env, system_template=system, instance_template="{{task}}",
            step_limit=V5_ROLE_STEP_LIMIT, cost_limit=0, wall_time_limit_seconds=int(timeout_sec or 1800),
        )


class MiniV63Agent(MiniV62Agent):
    workflow_version = "6.3"
    workflow_config_type = V63Config

    @staticmethod
    def name():
        return "mini-v63-review-repair"

    def _make_role_runner(self, **kwargs):
        return MiniV63RoleRunner(**kwargs)
