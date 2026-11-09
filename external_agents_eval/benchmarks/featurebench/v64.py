"""featurebench_v6.4_woG: FeatureBench adapters around the unchanged V6.4 review path.

Benchmark-only differences from the DeepSWE V6.4 run:
- FeatureBench's task baseline is /testbed, so the review agent and the Repair
  system prompt use /testbed instead of DeepSWE's /app;
- this experiment runs without the build gate G (``build_gate_enabled=False``);
- A is the unchanged V6.3 Mini A with the FeatureBench /testbed path alias, and the
  ledger-only entry stops right after A, before any Solver call.
Prompts (v5-6), controller (v5-5), models and budgets are inherited unchanged.
"""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from external_agents_eval.agents.mini_ft_review import FrozenMiniReviewAgent, FrozenMiniReviewRunner
from external_agents_eval.agents.mini_v62 import MiniWorkspaceManager
from external_agents_eval.agents.mini_v63 import REVIEW_ROLES
from external_agents_eval.agents.mini_v64 import MiniV64ReviewAgent, MiniV64ReviewRunner
from external_agents_eval.benchmarks.featurebench.adapter import FeatureBenchEnvironment
from external_agents_eval.benchmarks.featurebench.agents import (
    FeatureBenchPiFileEnvironment,
    FeatureBenchV63Agent,
)
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v64_config import V64Config
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller

EXPERIMENT_ID = "featurebench_v6.4_woG"
FEATUREBENCH_BASELINE = "/testbed"
ROLE_WORKTREES = ("solver", "spec_review", "patch_review", "repair", "summary")


@dataclass(frozen=True)
class V64WoGConfig(V64Config):
    """V6.4 prompts and controller with the build gate switched off."""

    build_gate_enabled: bool = False


EDITABLE_HELPER = "/external_eval/featurebench_editable_assets.py"


def in_baseline(directory: str) -> bool:
    return directory == FEATUREBENCH_BASELINE or directory.startswith(FEATUREBENCH_BASELINE + "/")


class FeatureBenchV64Environment(FeatureBenchEnvironment):
    """FeatureBench bridge whose role-worktree commands bypass editable redirects.

    A meson-python editable finder always maps its package to /testbed. Commands that
    run in a role worktree set that finder's own skip marker, so the worktree's copy
    (with its seeded build outputs) is imported; /testbed commands are unchanged.
    """

    def __init__(self, container, *, instance_id: str):
        super().__init__(container, instance_id=instance_id)
        self.editable_skip: dict[str, list[str]] = {}

    def set_editable(self, scan: dict[str, Any]) -> None:
        skip: dict[str, list[str]] = {}
        for build in scan.get("builds", []):
            skip.setdefault(build["marker"], []).append(build["build_path"])
        self.editable_skip = skip

    def exec_sync(self, command, *, cwd=None, timeout_sec=None, **kwargs):
        directory = cwd or FEATUREBENCH_BASELINE
        if self.editable_skip and not in_baseline(directory):
            exports = "".join(
                f"export {marker}={shlex.quote(':'.join(paths))}\n"
                for marker, paths in sorted(self.editable_skip.items())
            )
            command = exports + command
        return super().exec_sync(command, cwd=cwd, timeout_sec=timeout_sec, **kwargs)


class FeatureBenchWorkspaceManager(MiniWorkspaceManager):
    """Mini worktrees plus the build outputs of meson-python editable installs."""

    editable: dict[str, Any] = {"builds": []}

    async def prepare(self, *, base_commit):
        await self.environment.upload_file(Path(__file__).with_name("editable_assets.py"), EDITABLE_HELPER)
        result = await self.environment.exec(
            f"python {EDITABLE_HELPER} scan --repository {shlex.quote(self.repository)} --rebuild",
            cwd=self.repository, timeout_sec=1800,
        )
        if result.return_code:
            raise RuntimeError("Editable-install scan failed: " + (result.stdout + result.stderr)[-2000:])
        self.editable = json.loads(result.stdout.splitlines()[-1])
        (self.transfer_dir.parent / "editable-assets.json").parent.mkdir(parents=True, exist_ok=True)
        (self.transfer_dir.parent / "editable-assets.json").write_text(json.dumps(self.editable, indent=2))
        set_editable = getattr(self.environment, "set_editable", None)
        if self.editable["builds"]:
            if set_editable is None:
                raise RuntimeError("This bridge cannot bypass editable-install redirects")
            set_editable(self.editable)
        return await super().prepare(base_commit=base_commit)

    async def _seed_runtime(self, worktree):
        await super()._seed_runtime(worktree)
        if self.editable["builds"]:
            result = await self.environment.exec(
                f"python {EDITABLE_HELPER} seed --worktree {shlex.quote(worktree)}",
                cwd=self.repository, timeout_sec=600,
            )
            if result.return_code:
                raise RuntimeError("Editable build-output seeding failed: "
                                   + (result.stdout + result.stderr)[-2000:])


class FeatureBenchV64ReviewRunner(MiniV64ReviewRunner):
    baseline_root = FEATUREBENCH_BASELINE

    def _mini_system(self, *, role, tools, schemas):
        if role in REVIEW_ROLES or role in {RoleName.REPAIR, RoleName.V5_SUMMARY_REVIEWER}:
            return super()._mini_system(role=role, tools=tools, schemas=schemas)
        # Inherited roles keep their original /testbed wording, which is FeatureBench's
        # real baseline path; DeepSWE's /app substitution does not apply here.
        return super(FrozenMiniReviewRunner, self)._mini_system(role=role, tools=tools, schemas=schemas)


class FeatureBenchV64ReviewAgent(MiniV64ReviewAgent):
    workdir = FEATUREBENCH_BASELINE
    workflow_config_type = V64WoGConfig

    def __init__(self, *args, build_gate=None, **kwargs):
        if build_gate is not None:
            raise ValueError(EXPERIMENT_ID + " runs without the build gate")
        self.build_gate = None
        FrozenMiniReviewAgent.__init__(self, *args, **kwargs)

    @staticmethod
    def name():
        return "featurebench-v64-wog-mini-review"

    def _make_role_runner(self, **kwargs):
        return FeatureBenchV64ReviewRunner(**kwargs)

    def _make_workspace(self, **kwargs):
        return FeatureBenchWorkspaceManager(**kwargs)

    def _make_controller(self, **kwargs):
        return V6Controller(**kwargs, build_gate=None)


class LedgerReady(Exception):
    """Raised by the ledger-only entry right after A; no Solver call follows."""

    def __init__(self, ledger: dict[str, Any] | None, quality: str, warnings: list, error: Any):
        super().__init__("A finished; the ledger-only entry stops before Solver")
        self.ledger = ledger
        self.quality = quality
        self.warnings = list(warnings)
        self.error = error


class FeatureBenchLedgerAgent(FeatureBenchV63Agent):
    """Run the unchanged V6.3 Mini A on FeatureBench and stop before Solver."""

    @staticmethod
    def name():
        return "featurebench-v63-a-only"

    def _make_workspace(self, **kwargs):
        return FeatureBenchWorkspaceManager(**kwargs)

    def _make_controller(self, **kwargs):
        controller = super()._make_controller(**kwargs)
        prepare = controller.prepare_ledger

        async def prepare_then_stop(*args, **options):
            ledger = await prepare(*args, **options)
            raise LedgerReady(ledger, controller.ledger_quality, list(controller.ledger_warnings),
                              controller.ledger_error)

        controller.prepare_ledger = prepare_then_stop
        return controller


IMPORT_PROBE = (
    "import importlib, json, os, sys\n"
    "module = importlib.import_module(sys.argv[1])\n"
    "origin = getattr(module, '__file__', None) or list(getattr(module, '__path__', []))[0]\n"
    "path = os.path.realpath(origin)\n"
    "root = os.path.realpath(os.getcwd()) + '/'\n"
    "top = sys.argv[1].split('.')[0]\n"
    "ext = [os.path.realpath(m.__file__) for m in list(sys.modules.values())\n"
    "       if getattr(m, '__name__', '').split('.')[0] == top and getattr(m, '__file__', None)\n"
    "       and m.__file__.endswith(('.so', '.pyd'))]\n"
    "print(json.dumps({'module': path, 'inside_worktree': path.startswith(root)\n"
    "                  and all(p.startswith(root) for p in ext), 'extensions': len(ext),\n"
    "                  'extensions_outside': [p for p in ext if not p.startswith(root)][:5],\n"
    "                  'python': sys.executable}))\n"
)


def pick_probe_file(tracked: list[str], library: str | None) -> str | None:
    preferred = []
    if library:
        top = library.split(".")[0]
        preferred = [f"{top}/__init__.py", f"src/{top}/__init__.py"]
    for path in preferred:
        if path in tracked:
            return path
    python = sorted(path for path in tracked if path.endswith(".py"))
    if python:
        return python[0]
    return sorted(tracked)[0] if tracked else None


async def check_role_worktrees(environment, *, base_commit: str, library: str | None,
                               output: Path) -> dict[str, Any]:
    """Zero-API check: every role worktree imports its own tree; A's Pi reads alias /testbed."""
    from external_agents_eval.agents.pi_file_tools import PI_EVIDENCE_PROTOCOL, prepare_tools
    from external_agents_eval.workflows.solve_review_repair.ft_evidence import observed_evidence

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    workspace = FeatureBenchWorkspaceManager(environment=environment, repository=FEATUREBENCH_BASELINE,
                                             transfer_dir=output / "transfers")
    report: dict[str, Any] = {"library": library, "roles": {}, "passed": False}
    try:
        paths = await workspace.prepare(base_commit=base_commit)
        report["editable"] = workspace.editable
        status = await environment.exec("git status --porcelain --untracked-files=all",
                                        cwd=paths.spec_review, timeout_sec=300)
        report["seeded_worktree_clean"] = not status.stdout.strip()
        for role in ROLE_WORKTREES:
            cwd = getattr(paths, role)
            if library is None:
                report["roles"][role] = {"worktree": cwd, "import_check": "not_applicable"}
                continue
            result = await environment.exec(
                "python -c " + shlex.quote(IMPORT_PROBE) + " " + shlex.quote(library),
                cwd=cwd, timeout_sec=600,
            )
            if result.return_code:
                report["roles"][role] = {"worktree": cwd, "error": (result.stdout + result.stderr)[-3000:]}
                continue
            report["roles"][role] = {"worktree": cwd, **json.loads(result.stdout.splitlines()[-1])}
        listing = await environment.exec("git ls-files", cwd=paths.spec_review, timeout_sec=300)
        tracked = [line for line in listing.stdout.splitlines() if line]
        probe_file = pick_probe_file(tracked, library)
        report["tracked_files"] = len(tracked)
        if probe_file is None:
            report["pi_read"] = {"status": "no_tracked_files"}
        else:
            await prepare_tools(environment)
            tools = FeatureBenchPiFileEnvironment(environment, paths.spec_review)
            for index, path in enumerate([probe_file, FEATUREBENCH_BASELINE + "/" + probe_file]):
                result = tools.execute({"name": "read", "arguments": {"path": path, "limit": 20},
                                        "tool_call_id": f"preflight-read-{index}"})
                if result["returncode"]:
                    raise RuntimeError("Pi read failed: " + result["output"][-2000:])
            steps = []
            for index, (name, arguments, result) in enumerate(tools.events):
                call_id = f"preflight-read-{index}"
                steps.append({
                    "step_id": index + 1,
                    "tool_calls": [{"tool_call_id": call_id, "function_name": name, "arguments": arguments}],
                    "observation": {"results": [{"source_call_id": call_id, "content": result["output"],
                                                  "exit_code": result["returncode"]}]},
                })
            evidence = await observed_evidence(
                {"evidence_protocol": PI_EVIDENCE_PROTOCOL, "steps": steps}, task_id="preflight",
                base_root=paths.spec_review, base_commit=base_commit, workspace=workspace,
            )
            report["pi_read"] = {
                "file": probe_file, "read_provenance": len(evidence["read_provenance"]),
                "skipped_reads": len(evidence["skipped_reads"]), "evidence_files": len(evidence["evidence"]),
                "status": "ok" if (not evidence["skipped_reads"] and len(evidence["read_provenance"]) == 2
                                   and evidence["evidence"]) else "lost_read",
            }
    finally:
        await workspace.cleanup()
    imports_ok = all(item.get("import_check") == "not_applicable" or item.get("inside_worktree") is True
                     for item in report["roles"].values()) and len(report["roles"]) == len(ROLE_WORKTREES)
    report["passed"] = bool(imports_ok and report.get("seeded_worktree_clean")
                            and report.get("pi_read", {}).get("status") in {"ok", "no_tracked_files"})
    (output / "role-worktrees.json").write_text(json.dumps(report, indent=2))
    return report
