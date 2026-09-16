import json
import subprocess
import sys
from pathlib import Path

from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    load_frozen_few_shots,
)
from external_agents_eval.workflows.solve_review_repair.config import (
    DEFAULT_PROMPT_DIR,
    DEFAULT_SCHEMA_DIR,
    WorkflowConfig,
)
from external_agents_eval.workflows.solve_review_repair.prompt_renderer import PromptRenderer
from external_agents_eval.workflows.solve_review_repair.validators import ContractValidator

ROOT = Path(__file__).resolve().parents[1]
V3_FEW_SHOTS = ROOT / "few_shots/source_grounded_v3"
V4_FEW_SHOTS = ROOT / "few_shots/source_grounded_v4"


def _config(prompt_version: str) -> WorkflowConfig:
    return WorkflowConfig(
        contract_version="3.0",
        experiment_arm="E2",
        max_obligations=64,
        prompt_version=prompt_version,
    )


def test_v4_replaces_only_the_stage_b_prompt_bundle():
    v3 = _config("3.0")
    v4 = _config("4.0")

    assert v3.prompt_templates == {
        "build_obligations": "build_obligations_v3.md",
        "assess_patch": "assess_patch_v3.md",
        "repair_once": "repair_once_v3.md",
        "correct_json": "correct_json_v3.md",
    }
    assert v4.prompt_templates == {
        **v3.prompt_templates,
        "assess_patch": "assess_patch_v4.md",
    }
    v4.validate()


def test_v4_few_shots_change_only_codec_stage_b_and_provenance():
    expected_changes = {
        "provenance.json",
        "synthetic-format-codec/stage_b_input_summary.json",
        "synthetic-format-codec/stage_b_output.json",
    }
    v3_paths = {
        path.relative_to(V3_FEW_SHOTS).as_posix(): path
        for path in V3_FEW_SHOTS.rglob("*")
        if path.is_file()
    }
    v4_paths = {
        path.relative_to(V4_FEW_SHOTS).as_posix(): path
        for path in V4_FEW_SHOTS.rglob("*")
        if path.is_file()
    }

    assert set(v4_paths) == set(v3_paths)
    actual_changes = {
        name for name in v3_paths if v3_paths[name].read_bytes() != v4_paths[name].read_bytes()
    }
    assert actual_changes == expected_changes

    v3 = load_frozen_few_shots(V3_FEW_SHOTS)
    v4 = load_frozen_few_shots(V4_FEW_SHOTS)
    assert v3.prompt_version is None
    assert v4.contract_version == "3.0"
    assert v4.prompt_version == "4.0"
    assert [path.read_bytes() for path in v4.stage_a] == [path.read_bytes() for path in v3.stage_a]
    assert [path.read_bytes() for path in v4.repair] == [path.read_bytes() for path in v3.repair]


def test_v4_codec_assessment_uses_the_existing_v3_contract():
    case = V4_FEW_SHOTS / "synthetic-format-codec"
    ledger = json.loads((case / "stage_a_output.json").read_text())
    assessment = json.loads((case / "stage_b_output.json").read_text())
    validator = ContractValidator(DEFAULT_SCHEMA_DIR, contract_version="3.0")

    validator.validate_assessment(
        assessment,
        task_id="synthetic-format-codec",
        patch_sha256=assessment["patch_sha256"],
        ledger_sha256=assessment["ledger_sha256"],
        ledger=ledger,
    )


def test_v4_stage_b_prompt_and_examples_do_not_exceed_v3():
    renderer = PromptRenderer(DEFAULT_PROMPT_DIR)
    v3 = load_frozen_few_shots(V3_FEW_SHOTS)
    v4 = load_frozen_few_shots(V4_FEW_SHOTS)
    replacements = {
        "PATCH_ASSESSMENT_OUTPUT_SCHEMA": "{}",
        "ASSESS_PATCH_INPUT_JSON": "{}",
    }

    def rendered_length(template: str, paths: tuple[Path, ...]) -> int:
        prompt, _ = renderer.render(
            template,
            {
                **replacements,
                "STAGE_B_FEW_SHOT_EXAMPLES": renderer.load_few_shots(paths),
            },
        )
        return len(prompt)

    old_codec = v3.stage_b[2:4]
    new_codec = v4.stage_b[2:4]
    assert len(renderer.load_few_shots(new_codec)) <= len(renderer.load_few_shots(old_codec)) * 1.1
    assert rendered_length("assess_patch_v4.md", v4.stage_b) <= rendered_length(
        "assess_patch_v3.md", v3.stage_b
    )


def test_v4_codec_example_probe_distinguishes_candidate_and_repair(tmp_path):
    package = tmp_path / "src"
    tests = tmp_path / "tests"
    package.mkdir()
    tests.mkdir()
    (package / "__init__.py").write_text("")
    (tests / "test_codec.py").write_text(
        "from src.codec import decode, encode\n\n"
        "def test_default_payload_compatibility():\n"
        "    assert decode(encode('x')) == 'x'\n"
    )

    def write_codec(*, repaired: bool) -> None:
        lines = ["import json", "DEFAULT_VERSION = 1"]
        lines.extend([""] * (27 - len(lines)))
        lines.extend(
            [
                "def encode(value, format_version=DEFAULT_VERSION):",
                "    return json.dumps({'version': "
                + ("format_version" if repaired else "DEFAULT_VERSION")
                + ", 'value': value}).encode()",
            ]
        )
        lines.extend([""] * (41 - len(lines)))
        lines.extend(
            [
                "def decode(payload):",
                "    return json.loads(payload)['value']",
            ]
        )
        (package / "codec.py").write_text("\n".join(lines) + "\n")

    default_command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_codec.py",
        "-q",
    ]
    probe_command = [
        sys.executable,
        "-c",
        (
            "import json; from src.codec import encode; "
            "p=json.loads(encode('x', format_version=2)); "
            "assert p['version'] == 2, p"
        ),
    ]

    write_codec(repaired=False)
    assert subprocess.run(default_command, cwd=tmp_path, check=False).returncode == 0
    assert subprocess.run(probe_command, cwd=tmp_path, check=False).returncode != 0

    write_codec(repaired=True)
    assert subprocess.run(probe_command, cwd=tmp_path, check=False).returncode == 0
    assert subprocess.run(default_command, cwd=tmp_path, check=False).returncode == 0
