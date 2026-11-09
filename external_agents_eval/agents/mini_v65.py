"""V6.5 Mini review backend: review view (A), assumptions (C) and the D lane.

Relative to V6.4 this adds a workspace that builds and installs the review view,
the D reviewer (same file-delivery contract as I/R, role letter D) and the
tool-less assumption extractor X, whose session is never uploaded to the task
container. I/R/S/Repair system prompts are the V6.4 ones.
"""
from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

from external_agents_eval.agents.mini_review_files import FILE_CONTRACT_V64, controller_state
from external_agents_eval.agents.mini_v62 import MiniBashEnvironment, MiniWorkspaceManager
from external_agents_eval.agents.mini_v63 import MiniFileReviewEnvironment
from external_agents_eval.agents.mini_v64 import (
    REVIEW_SYSTEM_V64,
    MiniV64ReviewAgent,
    MiniV64ReviewRunner,
)
from external_agents_eval.workflows.solve_review_repair import review_view
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes
from external_agents_eval.workflows.solve_review_repair.v5_prompts import (
    OUTPUT_CONTRACT_MARKER,
    REVIEW_BATCH_SCHEMA,
)
from external_agents_eval.workflows.solve_review_repair.v65_config import V65Config
from external_agents_eval.workflows.solve_review_repair.v65_controller import (
    V65Controller,
    V65Role,
)

VIEW_WORKTREE = "v65-review-view"
VIEW_HELPER = "/external_eval/v65_review_view.py"

ASSUMPTION_SYSTEM_V65 = """You are an analyst reading a code change. Filesystem and shell
tools are disabled for this stage; everything you need is in the task message. Every
response must contain exactly one Bash tool call. When finished, submit the complete
JSON object with: submit_json '<complete JSON>'. submit_json is recognized by the
harness, not an installed program; its single-quoted argument must be the literal
JSON text.
"""

_ROLE_LINE = "'# Review note: <ROLE>_SCOPE' (ROLE is I or R)"
if FILE_CONTRACT_V64.count(_ROLE_LINE) != 1:
    raise RuntimeError("V6.4 file contract changed; update the V6.5 D contract")
FILE_CONTRACT_V65_D = FILE_CONTRACT_V64.replace(
    _ROLE_LINE, "'# Review note: D_SCOPE'"
) + """
For D, the Assessment block's expected states the reading the evidence supports,
observed states the reading the candidate implements with the discriminating probe's
decisive output, and reason names the evidence class (a), (b) or (c) that decided it.
"""


def adapt_prompt_v65_d(prompt: str) -> str:
    """Replace the D prompt's JSON OUTPUT CONTRACT with the D file contract."""
    controller_state(prompt)
    if prompt.count(OUTPUT_CONTRACT_MARKER) != 1:
        raise ValueError("Expected exactly one OUTPUT CONTRACT in D Reviewer input")
    body = prompt.split(OUTPUT_CONTRACT_MARKER, 1)[0]
    definitions = {
        key: value for key, value in REVIEW_BATCH_SCHEMA["$defs"].items()
        if key not in {"Assessment", "ReviewBatch"}
    }
    return (body + "\n" + FILE_CONTRACT_V65_D + "\nFIELD DEFINITIONS\n"
            + json.dumps({"$defs": definitions}, ensure_ascii=False, indent=2) + "\n")


class MiniV65WorkspaceManager(MiniWorkspaceManager):
    """Build the review view in a scratch worktree and install it for reviewers."""

    _view_worktree: str | None = None
    _view_helper_ready = False

    async def _remove_input(self, name: str) -> str | None:
        path = f"{self.INPUT_ROOT}/{name}"
        exists = await self.environment.exec(f"test -e {shlex.quote(path)}")
        if exists.return_code != 0:
            return None
        await self._checked(f"rm -f -- {shlex.quote(path)}")
        return path

    async def build_review_view(self, *, base_commit, baseline_patch, candidate_patch, level):
        if not self._view_helper_ready:
            await self.environment.upload_file(Path(review_view.__file__), VIEW_HELPER)
            self._view_helper_ready = True
        if self._view_worktree is None:
            self._view_worktree = await self.prepare_additional_worktree(
                name=VIEW_WORKTREE, base_commit=base_commit)
        worktree = self._view_worktree
        await self.restore_reviewer_worktree(
            worktree=worktree, base_commit=base_commit, baseline_patch=baseline_patch)
        await self.apply_patch(worktree=worktree, patch=candidate_patch)
        patch_path = await self.materialize_file(
            name=f"v65-view-input-{sha256_bytes(candidate_patch)[:16]}.patch",
            payload=candidate_patch,
        )
        result = await self._checked(
            "python " + " ".join(shlex.quote(part) for part in (
                VIEW_HELPER, "--worktree", worktree, "--patch", patch_path, "--level", level)),
            cwd=worktree,
        )
        manifest = json.loads((result.stdout or "").strip().splitlines()[-1])
        view = await self.capture_patch(
            worktree=worktree, base_commit=base_commit, baseline_patch=baseline_patch)
        return view, manifest

    async def discard_review_view(self, *, candidate_patch):
        if self._view_worktree is not None:
            await self.discard_worktree(worktree=self._view_worktree)
            self._view_worktree = None
        await self._checked("git worktree prune", cwd=self.repository)
        await self._remove_input(f"v65-view-input-{sha256_bytes(candidate_patch)[:16]}.patch")

    async def reset_review_worktree(self, *, worktree, base_commit, baseline_patch, patch):
        await self.restore_reviewer_worktree(
            worktree=worktree, base_commit=base_commit, baseline_patch=baseline_patch)
        await self.apply_patch(worktree=worktree, patch=patch)

    async def prepare_review_worktree(self, *, name, base_commit, baseline_patch, patch):
        worktree = await self.prepare_additional_worktree(name=name, base_commit=base_commit)
        if baseline_patch:
            await self.apply_patch(worktree=worktree, patch=baseline_patch)
        await self.apply_patch(worktree=worktree, patch=patch)
        return worktree

    async def remove_candidate_inputs(self, *, candidate_patch):
        # apply_patch leaves apply-<sha16>.patch behind; delete every raw-P0 copy.
        digest = sha256_bytes(candidate_patch)[:16]
        removed = []
        for name in (f"apply-{digest}.patch", f"v5-p0-{digest}.patch",
                     f"v65-view-input-{digest}.patch"):
            path = await self._remove_input(name)
            if path is not None:
                removed.append(path)
        return removed

    async def prune_unreachable_objects(self):
        # Temporary-index captures left candidate blobs in the shared object store.
        def count(output):
            values = {}
            for line in (output or "").splitlines():
                key, _, value = line.partition(":")
                values[key.strip()] = value.strip()
            return values

        await self._checked("git worktree prune", cwd=self.repository)
        before = await self._checked("git count-objects -v", cwd=self.repository)
        await self._checked("git prune --expire=now", cwd=self.repository)
        after = await self._checked("git count-objects -v", cwd=self.repository)
        return {"before": count(before.stdout), "after": count(after.stdout)}


class MiniV65ReviewRunner(MiniV64ReviewRunner):
    async def prepare_review_catalog(self, *, role, catalog):
        """Publish the D catalog like the inherited I/R one (which accepts only I/R)."""
        if role != "D":
            return await super().prepare_review_catalog(role=role, catalog=catalog)
        payload = (json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        digest = sha256_bytes(payload)
        remote = f"/external_eval/v5_resources/catalog-D-{digest[:16]}.json"
        exists = await self.environment.exec(f"test -f {shlex.quote(remote)}")
        if exists.return_code == 0:
            return remote
        host = self.logs_root / "v5" / "catalogs" / f"{digest}.json"
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_bytes(payload)
        await self.environment.exec("mkdir -p /external_eval/v5_resources")
        await self.environment.upload_file(host, remote)
        await self.environment.exec(f"chmod 0444 {shlex.quote(remote)}")
        return remote

    async def run_role(self, *, role, prompt, worktree, tools, timeout_sec,
                       artifact_prefix, session_id=None, session_dir=None,
                       session_file=None, no_compaction=False, invocation_id=None):
        if role == V65Role.ASSUMPTION_EXTRACTOR:
            if tuple(tools) != ():
                raise ValueError("The V6.5 assumption extractor must be tool-less")
            # Never upload this session: it holds the text withheld from reviewers.
            return await asyncio.to_thread(
                self._run_mini, role, prompt, worktree, timeout_sec, artifact_prefix,
                None, (), None,
            )
        if role == V65Role.DISAMBIGUATION_REVIEWER:
            adapted = adapt_prompt_v65_d(prompt)
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

    def _make_mini_environment(self, *, role, prompt, worktree, tools, schemas, host):
        if role == V65Role.DISAMBIGUATION_REVIEWER:
            return MiniFileReviewEnvironment(
                self.environment, worktree, role="D", state=controller_state(prompt),
                archive=host / "review-records", redact=self.redact,
            )
        if role == V65Role.ASSUMPTION_EXTRACTOR:
            return MiniBashEnvironment(self.environment, worktree, tools=())
        return super()._make_mini_environment(
            role=role, prompt=prompt, worktree=worktree, tools=tools, schemas=schemas, host=host,
        )

    def _mini_system(self, *, role, tools, schemas):
        if role == V65Role.DISAMBIGUATION_REVIEWER:
            return REVIEW_SYSTEM_V64
        if role == V65Role.ASSUMPTION_EXTRACTOR:
            return ASSUMPTION_SYSTEM_V65
        return super()._mini_system(role=role, tools=tools, schemas=schemas)


class MiniV65ReviewAgent(MiniV64ReviewAgent):
    workflow_version = "6.5"
    workflow_config_type = V65Config

    @staticmethod
    def name():
        return "v65-mini-review"

    def _make_role_runner(self, **kwargs):
        return MiniV65ReviewRunner(**kwargs)

    def _make_workspace(self, **kwargs):
        return MiniV65WorkspaceManager(**kwargs)

    def _make_controller(self, **kwargs):
        return V65Controller(**kwargs, build_gate=self.build_gate)
