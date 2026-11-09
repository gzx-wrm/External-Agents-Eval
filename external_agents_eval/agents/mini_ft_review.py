"""Review a frozen native-Mini P0 with the existing V6 controller and Mini backend.

No A/Solver call is permitted here. The new experiment imports their frozen
outputs and changes only the downstream review path.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from external_agents_eval.agents.mini_v63 import MiniV63Agent, MiniV63RoleRunner
from external_agents_eval.workflows.solve_review_repair.artifacts import (
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller
from external_agents_eval.workflows.solve_review_repair.v6_fork import write_stage

# Native Mini exports HEAD -> candidate, including image-only baseline changes.
# V6 expects environment-baseline -> candidate. Temporary indexes preserve all
# tracked/untracked file content already present in the frozen native patch.
NORMALIZE_SCRIPT = r"""import json, os, pathlib, subprocess, sys, tempfile
base, baseline_path, candidate_path, output_path = sys.argv[1:]
def git(*args, env=None, data=None):
    return subprocess.check_output(['git', *args], env=env, input=data)
with tempfile.TemporaryDirectory(prefix='ft-m1-review-index-') as folder:
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = folder + '/index'
    def tree(start, patch):
        git('read-tree', start, env=env)
        if patch:
            git('apply', '--cached', '--binary', '--whitespace=nowarn', '-', env=env, data=patch)
        return git('write-tree', env=env).decode().strip()
    baseline = pathlib.Path(baseline_path).read_bytes()
    candidate = pathlib.Path(candidate_path).read_bytes()
    baseline_tree = tree(base, baseline)
    candidate_tree = tree(base, candidate)
    # V6 also hashes its environment patch. Canonicalize BOTH patch formats,
    # since native exports full-index headers and V6 capture_patch does not.
    canonical_baseline = git('diff', '--binary', '--no-ext-diff', base, baseline_tree, '--', '.')
    pathlib.Path(output_path + '.baseline').write_bytes(canonical_baseline)
    delta = git('diff', '--binary', '--no-ext-diff',
                baseline_tree, candidate_tree, '--', '.')
    reconstructed_tree = tree(baseline_tree, delta)
    if reconstructed_tree != candidate_tree:
        raise ValueError('P0 normalization changed the candidate source tree')
    pathlib.Path(output_path).write_bytes(delta)
    print(json.dumps({'baseline_tree': baseline_tree, 'candidate_tree': candidate_tree,
                      'reconstructed_tree': reconstructed_tree, 'candidate_unchanged': True}))
"""


def normalization_command(base: str, baseline_path: str, candidate_path: str, output_path: str):
    return "python -c " + shlex.quote(NORMALIZE_SCRIPT) + " " + " ".join(
        shlex.quote(value) for value in (base, baseline_path, candidate_path, output_path)
    )


def write_frozen_fork(
    root: Path, *, task_id: str, base_commit: str, baseline_patch: bytes,
    ledger: dict, candidate_patch: bytes, provenance: dict,
) -> Path:
    """Use the established stage-reuse protocol, without fabricating an A run."""
    source = root / "frozen"
    files = {
        source / "ledger.json": (json.dumps(ledger, ensure_ascii=False, indent=2) + "\n").encode(),
        source / "outputs/v6-p0.patch": candidate_patch,
        source / "provenance.json": (json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode(),
    }
    for path, data in files.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Frozen review input changed: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_stage(
        source, stage="Solver", task_id=task_id, base_commit=base_commit,
        baseline_sha256=sha256_bytes(baseline_patch), ledger_sha256=sha256_json(ledger),
        patch_sha256=sha256_bytes(candidate_patch), reused_from=provenance["m1_output"],
    )
    plan = root / "fork-plan.json"
    payload = {"task_id": task_id, "start_at": "I/R", "reuse": {
        "A": str((source / "ledger.json").resolve()), "Solver": str(source.resolve()),
    }}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if plan.exists() and plan.read_text() != encoded:
        raise ValueError("Frozen review fork plan changed")
    plan.write_text(encoded)
    return plan


class FrozenMiniReviewRunner(MiniV63RoleRunner):
    async def run_role(self, *, role, prompt, worktree, tools, timeout_sec,
                       artifact_prefix, session_id=None, session_dir=None,
                       session_file=None, no_compaction=False, invocation_id=None):
        # V6 inspects this signature before supplying session and invocation
        # bindings. Keep the parent interface explicit instead of hiding it in **kwargs.
        if role in {RoleName.SOLVER, RoleName.V5_SPECIFICATION_REVIEWER}:
            raise ValueError("ft_m1_review must reuse A and Solver; resampling is forbidden")
        return await super().run_role(
            role=role, prompt=prompt, worktree=worktree, tools=tools, timeout_sec=timeout_sec,
            artifact_prefix=artifact_prefix, session_id=session_id, session_dir=session_dir,
            session_file=session_file, no_compaction=no_compaction, invocation_id=invocation_id,
        )

    def _mini_system(self, **kwargs):
        # The inherited Mini backend was first used on SWE-bench /testbed.
        # DeepSWE's frozen baseline is /app; role guidance is otherwise identical.
        return super()._mini_system(**kwargs).replace("/testbed", "/app")


class FrozenMiniReviewAgent(MiniV63Agent):
    workdir = "/app"

    @staticmethod
    def name():
        return "ft-m1-mini-review"

    def _make_role_runner(self, **kwargs):
        return FrozenMiniReviewRunner(**kwargs)

    def _make_controller(self, **kwargs):
        # Reuse V6's imported-ledger path directly. No FT client or evidence
        # collection is needed, and review policy remains the existing V6 policy.
        return V6Controller(**kwargs)
