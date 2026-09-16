from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import RoleName

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
PROJECT_PACKAGE = PROJECT_ROOT / "external_agents_eval"
DEFAULT_SCHEMA_DIR = PACKAGE_DIR / "schemas"
DEFAULT_PROMPT_DIR = PACKAGE_DIR / "prompts"

V2_RUNTIME_IMPLEMENTATION_FILES = (
    PROJECT_PACKAGE / "cli.py",
    PROJECT_PACKAGE / "agents/pier/pi.py",
    PROJECT_PACKAGE / "agents/pier/pi_role_runner.py",
    PROJECT_PACKAGE / "agents/pier/pi_solve_review_repair.py",
    PROJECT_PACKAGE / "agents/pier/trajectory.py",
    PROJECT_PACKAGE / "benchmarks/deepswe/config.py",
    PROJECT_PACKAGE / "benchmarks/deepswe/pier_compat.py",
    PROJECT_PACKAGE / "benchmarks/deepswe/report.py",
    PROJECT_PACKAGE / "benchmarks/deepswe/reviewer_experiment_manifest.py",
    PROJECT_PACKAGE / "benchmarks/deepswe/runner.py",
)

READ_ONLY_TOOLS = ("read", "bash", "grep", "find", "ls")
CORRECTION_TOOLS = ("read", "grep", "find", "ls")
WRITE_TOOLS = ("read", "bash", "edit", "write", "grep", "find", "ls")

SUPPORTED_CONTRACT_VERSIONS = ("1.0", "2.0", "3.0")
SUPPORTED_EXPERIMENT_ARMS = ("E0", "E1", "E2")

PROMPT_BUNDLES = {
    "1.0": {
        "build_obligations": "build_obligations_v1.md",
        "assess_patch": "assess_patch_v1.md",
        "repair_once": "repair_once_v1.md",
        "correct_json": "correct_json_v1.md",
    },
    "2.0": {
        "build_obligations": "build_obligations_v2.md",
        "assess_patch": "assess_patch_v2.md",
        "repair_once": "repair_once_v2.md",
        "correct_json": "correct_json_v2.md",
    },
    "3.0": {
        "build_obligations": "build_obligations_v3.md",
        "assess_patch": "assess_patch_v3.md",
        "repair_once": "repair_once_v3.md",
        "correct_json": "correct_json_v3.md",
    },
}


@dataclass(frozen=True)
class WorkflowConfig:
    schema_dir: Path = DEFAULT_SCHEMA_DIR
    prompt_dir: Path = DEFAULT_PROMPT_DIR
    stage_a_few_shots: tuple[Path, ...] = ()
    stage_b_few_shots: tuple[Path, ...] = ()
    repair_few_shots: tuple[Path, ...] = ()
    few_shot_metadata: tuple[Path, ...] = ()
    implementation_files: tuple[Path, ...] = ()
    role_timeout_sec: float | None = None
    solver_timeout_sec: float = 10_800
    repair_timeout_sec: float = 10_800
    reviewer_timeout_sec: float = 3_600
    json_corrector_timeout_sec: float = 1_800
    max_obligations: int = 32
    max_criteria_per_obligation: int = 8
    max_json_corrections: int = 2
    contract_version: str = "1.0"
    source_policy_version: str = "1.0"
    experiment_arm: str = "E0"
    prompt_version: str = "1.3"
    controller_version: str = "1.3.0"
    model: str | None = None
    agent_version: str | None = None
    resume: bool = False

    @property
    def prompt_templates(self) -> dict[str, str]:
        templates = dict(PROMPT_BUNDLES[self.contract_version])
        if self.prompt_version == "4.0":
            templates["assess_patch"] = "assess_patch_v4.md"
        if self.experiment_arm == "E1" and self.contract_version == "2.0":
            templates["repair_once"] = "repair_once_v2_e1.md"
        return templates

    @property
    def repair_audit_required(self) -> bool:
        return self.experiment_arm == "E2"

    @property
    def implementation_paths(self) -> tuple[Path, ...]:
        workflow_files = tuple(sorted(self.schema_dir.parent.glob("*.py")))
        if self.contract_version == "1.0":
            return workflow_files
        ordered = (
            *workflow_files,
            *V2_RUNTIME_IMPLEMENTATION_FILES,
            *self.implementation_files,
        )
        return tuple(dict.fromkeys(path.resolve() for path in ordered))

    @staticmethod
    def implementation_label(path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return f"external:{path.name}"

    def tools_for(self, role: RoleName) -> tuple[str, ...]:
        if role in {
            RoleName.SPECIFICATION_REVIEWER,
            RoleName.PATCH_REVIEWER,
        }:
            return READ_ONLY_TOOLS
        if role == RoleName.JSON_CORRECTOR:
            return CORRECTION_TOOLS
        return WRITE_TOOLS

    def timeout_for(self, role: RoleName) -> float:
        if role == RoleName.SOLVER:
            return self.solver_timeout_sec
        if role == RoleName.REPAIR:
            return self.repair_timeout_sec
        if self.role_timeout_sec is not None:
            return self.role_timeout_sec
        if role == RoleName.JSON_CORRECTOR:
            return self.json_corrector_timeout_sec
        return self.reviewer_timeout_sec

    def validate(self) -> None:
        if self.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            raise ValueError(f"Unsupported workflow contract version: {self.contract_version}")
        if self.experiment_arm not in SUPPORTED_EXPERIMENT_ARMS:
            raise ValueError(f"Unsupported reviewer experiment arm: {self.experiment_arm}")
        allowed_contracts = {"1.0"} if self.experiment_arm == "E0" else {"2.0", "3.0"}
        if self.contract_version not in allowed_contracts:
            raise ValueError(
                f"Experiment arm {self.experiment_arm} requires contract in "
                f"{sorted(allowed_contracts)}"
            )
        if self.prompt_version == "4.0" and (
            self.contract_version != "3.0" or self.experiment_arm not in {"E1", "E2"}
        ):
            raise ValueError(
                "Reviewer prompt v4.0 requires contract v3.0 and experiment arm E1 or E2"
            )
        if self.contract_version == "2.0" and self.source_policy_version != "1.0":
            raise ValueError("Workflow contract v2.0 requires source policy version 1.0")
        expected_obligations = 64 if self.contract_version == "3.0" else 32
        if self.max_obligations != expected_obligations:
            raise ValueError(
                f"Workflow contract {self.contract_version} fixes max_obligations at "
                f"{expected_obligations}"
            )
        if self.contract_version == "2.0" and self.max_criteria_per_obligation != 8:
            raise ValueError("Schema v2.0 fixes max_criteria_per_obligation at 8")
        if self.max_json_corrections != 2:
            raise ValueError("The v1.1 workflow permits exactly two phased JSON corrections")
        timeouts = {
            "solver_timeout_sec": self.solver_timeout_sec,
            "repair_timeout_sec": self.repair_timeout_sec,
            "reviewer_timeout_sec": self.reviewer_timeout_sec,
            "json_corrector_timeout_sec": self.json_corrector_timeout_sec,
        }
        if self.role_timeout_sec is not None:
            timeouts["role_timeout_sec"] = self.role_timeout_sec
        invalid = sorted(name for name, value in timeouts.items() if value <= 0)
        if invalid:
            raise ValueError("Workflow role timeouts must be positive: " + ", ".join(invalid))
        required_prompts = set(self.prompt_templates.values())
        missing = sorted(
            name for name in required_prompts if not (self.prompt_dir / name).is_file()
        )
        if missing:
            raise FileNotFoundError("Missing workflow prompts: " + ", ".join(missing))
        for path in (
            *self.stage_a_few_shots,
            *self.stage_b_few_shots,
            *self.repair_few_shots,
            *self.few_shot_metadata,
            *self.implementation_paths,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen workflow artifact: {path}")
