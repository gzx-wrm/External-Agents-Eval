"""V6.6 review: the V6.3 frozen Mini review with check points per batch and in S.

I/R: the Mini review adapter inserts the check points of each batch's targets into that
round's prompt (before the file-delivery instructions, which the adapter would otherwise
replace), and a batch closes only when its notes carry complete checkpoint_results.
S: the summary prompt lists every check point and the decision order. Notes, the V6
controller, contracts, routing and budgets are FrozenMiniReviewAgent's.
v6.6.4: a Bash result in I/R/S keeps at most 30,000 characters (head and tail), so one
oversized command cannot overflow the model context.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from external_agents_eval.agents.mini_ft_review import FrozenMiniReviewAgent, FrozenMiniReviewRunner
from external_agents_eval.agents.mini_review_files import controller_state, metadata
from external_agents_eval.agents.mini_v62 import MiniBashEnvironment
from external_agents_eval.agents.mini_v63 import REVIEW_ROLES, MiniFileReviewEnvironment
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller
from external_agents_eval.workflows.solve_review_repair.v66_review_checks import (
    batch_section,
    checkpoint_result_errors,
)
from external_agents_eval.workflows.solve_review_repair.v66_review_prompts import summary_addendum

FILE_DELIVERY_MARKER = "V6.3 FILE DELIVERY"
# v6.6.4: one Bash result in I/R/S keeps at most this many characters (head + tail).
OUTPUT_CAP_CHARS = 30_000
OUTPUT_HEAD_CHARS = 15_000
OUTPUT_TAIL_CHARS = 15_000
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def cap_output(text: str) -> str:
    """Keep the head and tail of an oversized command result; never touch a submission."""
    if len(text) <= OUTPUT_CAP_CHARS or text.lstrip().startswith(SUBMIT_MARKER):
        return text
    omitted = len(text) - OUTPUT_HEAD_CHARS - OUTPUT_TAIL_CHARS
    notice = (f"\n\n[output truncated: {omitted} of {len(text)} characters omitted from the middle. "
              "Re-run a narrower command (head, sed -n, grep -c, or a smaller path) for the part you need.]\n\n")
    return text[:OUTPUT_HEAD_CHARS] + notice + text[-OUTPUT_TAIL_CHARS:]


def _append(prompt: str, addendum: str) -> str:
    return prompt.rstrip("\n") + "\n\n" + addendum if addendum else prompt


def batch_targets(prompt: str) -> set[str]:
    return {item["target_id"] for item in controller_state(prompt)["review_notes"]["target_paths"]}


def insert_batch_section(adapted: str, section: str) -> str:
    if not section:
        return adapted
    if adapted.count(FILE_DELIVERY_MARKER) != 1:
        raise ValueError("Expected exactly one V6.3 file-delivery section in the adapted review prompt")
    head, tail = adapted.split(FILE_DELIVERY_MARKER, 1)
    return head.rstrip("\n") + "\n\n" + section + "\n" + FILE_DELIVERY_MARKER + tail


class V66FileReviewEnvironment(MiniFileReviewEnvironment):
    """A valid V6.3 batch additionally needs complete checkpoint_results."""

    def __init__(self, *args: Any, checkpoints: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoints = checkpoints
        self.checkpoint_errors: list[list[str]] = []

    def _command_output(self, raw):
        return cap_output(super()._command_output(raw))

    def execute(self, action):
        result = super().execute(action)
        if self.batch is None:
            return result
        assessments = self.batch["assessments"]
        notes = {item["target_id"]: (self.archive / item["review_note_path"]).read_text(encoding="utf-8")
                 for item in assessments}
        errors = checkpoint_result_errors(notes, self.checkpoints,
                                          {item["target_id"] for item in assessments}, metadata)
        if errors:
            self.batch = None
            self.checkpoint_errors.append(errors)
            message = "Fix current files before continuing: checkpoint results are incomplete: " + "; ".join(errors[:20])
            result["output"] += f"\n[V6.6 checkpoint results] {message}\n"
            (self.archive.parent / "checkpoint-results-errors.json").write_text(
                json.dumps(self.checkpoint_errors, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


class V66SummaryBashEnvironment(MiniBashEnvironment):
    """S keeps the V6.3 Bash environment; only an oversized command result is bounded."""

    def _command_output(self, raw):
        return cap_output(super()._command_output(raw))


class V66FrozenReviewRunner(FrozenMiniReviewRunner):
    v66_checkpoints: list[dict[str, Any]] = []

    def _adapt_review_prompt(self, prompt):
        adapted = super()._adapt_review_prompt(prompt)
        return insert_batch_section(adapted, batch_section(self.v66_checkpoints, batch_targets(prompt)))

    def _make_mini_environment(self, *, role, prompt, worktree, tools, schemas, host):
        if role in REVIEW_ROLES:
            return V66FileReviewEnvironment(
                self.environment, worktree, role=REVIEW_ROLES[role],
                state=controller_state(prompt), archive=host / "review-records", redact=self.redact,
                checkpoints=self.v66_checkpoints,
            )
        environment = super()._make_mini_environment(
            role=role, prompt=prompt, worktree=worktree, tools=tools, schemas=schemas, host=host,
        )
        if role == RoleName.V5_SUMMARY_REVIEWER and type(environment) is MiniBashEnvironment:
            return V66SummaryBashEnvironment(self.environment, worktree, tools=tools)
        return environment


class V66ReviewController(V6Controller):
    checkpoints: list[dict[str, Any]] = []

    def _summary_prompt(self, summary_input: dict[str, Any], **kwargs: Any) -> str:
        return _append(super()._summary_prompt(summary_input, **kwargs),
                       summary_addendum(self.checkpoints))


class V66MiniReviewAgent(FrozenMiniReviewAgent):
    def __init__(self, *args: Any, checkpoints: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.v66_checkpoints = json.loads(Path(checkpoints).read_text(encoding="utf-8"))

    @staticmethod
    def name() -> str:
        return "v6.6-mini-review"

    def _make_role_runner(self, **kwargs):
        runner = V66FrozenReviewRunner(**kwargs)
        runner.v66_checkpoints = list(self.v66_checkpoints)
        return runner

    def _make_controller(self, **kwargs: Any) -> V66ReviewController:
        controller = V66ReviewController(**kwargs)
        controller.checkpoints = list(self.v66_checkpoints)
        return controller
