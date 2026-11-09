"""V6.4 Mini review backend over the frozen-P0 V6 controller path.

Changes relative to V6.3 are limited to prompt presentation and the build gate:
role-specific system prompts for I/R, S and Repair, and one coherent file
contract for I/R (record validation is unchanged). A and Solver stay frozen.
"""
from __future__ import annotations

from external_agents_eval.agents.mini_ft_review import FrozenMiniReviewAgent, FrozenMiniReviewRunner
from external_agents_eval.agents.mini_review_files import adapt_prompt_v64
from external_agents_eval.agents.mini_v63 import REVIEW_ROLES
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v64_config import V64Config
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller

REVIEW_SYSTEM_V64 = """You are a coding reviewer operating through the native Bash tool.
Your assigned working directory is {{cwd}}. Follow your role and the authoritative
Controller state. Do not edit tracked source or commit changes. Use the current
$REVIEW_OUTPUT_ROOT for review records, probes and logs. The sandbox is offline;
do not access remote solutions, hidden tests, grading outputs or other reviewers.
Execute one Bash action per response. Deliver your review as files, as specified
under OUTPUT CONTRACT (files), not through a final answer. No submit_json or finish
tool is needed: the adapter validates the files after each command and hands a valid
closed batch to the Controller automatically.
"""

REPAIR_SYSTEM_V64 = """You are a repair engineer using the Bash tool in {{cwd}}, which holds the
repository with the candidate patch P0 applied. Edit and test only in this working
tree; other paths are read-only, and {baseline} is the original baseline: do not edit
it or run candidate tests there. The sandbox is offline. Every response must contain
exactly one Bash tool call; never answer in plain text. Your deliverable is the state
of this working tree; do not commit. When finished, run a command whose first output
line is exactly COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, followed by a short summary of
what you changed and checked.
"""

SUMMARY_SYSTEM_V64 = """You are a summary reviewer using the Bash tool in {{cwd}} to read the
supplied review records, artifacts and repository. Do not edit repository or review
files and do not run new tests; scratch files under /tmp are fine. The sandbox is
offline. Every response must contain exactly one Bash tool call; never answer in plain
text. When finished, submit the complete JSON object in either way:
- run a command whose first output line is exactly COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT,
  followed by the JSON, for example:
  printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; cat /tmp/summary.json
- or run submit_json '<complete JSON>'. submit_json is recognized by the harness, not an
  installed program; its single-quoted argument must be the literal JSON text.
"""


class MiniV64ReviewRunner(FrozenMiniReviewRunner):
    baseline_root = "/app"

    def _adapt_review_prompt(self, prompt):
        return adapt_prompt_v64(prompt)

    def _mini_system(self, *, role, tools, schemas):
        if role in REVIEW_ROLES:
            return REVIEW_SYSTEM_V64
        if role == RoleName.REPAIR:
            return REPAIR_SYSTEM_V64.replace("{baseline}", self.baseline_root)
        if role == RoleName.V5_SUMMARY_REVIEWER:
            return SUMMARY_SYSTEM_V64
        return super()._mini_system(role=role, tools=tools, schemas=schemas)


class MiniV64ReviewAgent(FrozenMiniReviewAgent):
    workflow_version = "6.4"
    workflow_config_type = V64Config

    def __init__(self, *args, build_gate=None, **kwargs):
        if build_gate is None:
            raise ValueError("V6.4 review requires a build gate executor")
        self.build_gate = build_gate
        super().__init__(*args, **kwargs)

    @staticmethod
    def name():
        return "v64-mini-review"

    def _make_role_runner(self, **kwargs):
        return MiniV64ReviewRunner(**kwargs)

    def _make_controller(self, **kwargs):
        return V6Controller(**kwargs, build_gate=self.build_gate)
