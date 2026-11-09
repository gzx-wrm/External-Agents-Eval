"""Run the E9 A-only evaluation (training set or E6 validation+test) with online loop detection.

Same service, sampling, controller and fail-fast loop detector as E8; the
adapter is merged into the pinned base first, as in E8_1.  The system prompt is
the baseline E6/E7 prompt (no E8 repeat guard), so the comparison with E6/E7
isolates the training data.  ``--split heldout17`` runs the E6 validation (3)
and test (14) tasks with the same task inputs the E6 held-out runs used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path

from a_agent_finetune import run_e8_base_train13 as base

MERGE_ROOT = "/root/autodl-tmp/e9/inference/merge"
MERGE_SCRIPT = "/root/autodl-tmp/e9/inference/merge_for_inference.py"
LOCAL_MERGE_SCRIPT = base.HERE / "data/e7_normal13_recovery4_v1/provenance/merge_e7_for_inference.py"
E9_MATERIALS = "/root/autodl-tmp/e9/e9_normal13_rationale_v1"
RAW_BASE = base.REMOTE_MODEL
CHECKPOINTS = {
    "epoch-1": "/root/autodl-tmp/e9/runs/e9-normal13-rationale-20261013-r1/training/checkpoint-epoch-1/adapter",
    "epoch-2": "/root/autodl-tmp/e9/runs/e9-continue-epoch2-3-20261013-r1/training/checkpoint-epoch-2/adapter",
    "epoch-3": "/root/autodl-tmp/e9/runs/e9-continue-epoch2-3-20261013-r1/training/checkpoint-epoch-3/adapter",
    **{f"lr1e4-epoch-{epoch}": f"/root/autodl-tmp/e9/runs/e9-lr1e4-ep3-20261013-r1/training/checkpoint-epoch-{epoch}/adapter"
       for epoch in (1, 2, 3)},
    **{f"e9_1-lr1e4-epoch-{epoch}":
       f"/root/autodl-tmp/e9/runs/e9_1-lr1e4-ep3-20261013-r1/training/checkpoint-epoch-{epoch}/adapter"
       for epoch in (1, 2, 3)},
}
CHECKPOINTS |= {f"e10-lr1e4-epoch-{epoch}":
                f"/root/autodl-tmp/e10/runs/e10-train60-lr1e4-ep3-20261016-r1/training/checkpoint-epoch-{epoch}/adapter"
                for epoch in (1, 2, 3)}
E9_1_MATERIALS = "/root/autodl-tmp/e9/e9_1_normal13_rationale_closing_v1"
E10_MATERIALS = "/root/autodl-tmp/e10/e10_train60_rationale_closing_v1"
MERGE_PYTHON = "/root/autodl-tmp/g1/venv/bin/python"
# A800 instances; the E10 instance merges with the training venv (the vLLM venv has no peft).
INSTANCES = {
    "a800-57563": {"port": "57563", "base": base.REMOTE_MODEL, "vllm": base.VLLM, "cache": base.CACHE_ROOT,
                   "merge_python": MERGE_PYTHON, "work": "/root/autodl-tmp/e9"},
    "a800-46431": {"port": "46431",
                   "base": "/root/autodl-tmp/a-agent-smoke/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                           "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                   "vllm": "/root/autodl-tmp/a-agent-lora-inference/venv/bin/vllm",
                   "cache": "/root/autodl-tmp/a-agent-lora-inference",
                   "merge_python": "/root/a-agent-lora-smoke/venv/bin/python", "work": "/root/autodl-tmp/e10",
                   # This venv's torchao does not match its torch; importing unsloth first (as the training
                   # runner and this instance's earlier merges do) avoids transformers loading torchao.
                   "merge_preimport": "unsloth"},
}
MERGE_PREIMPORT = None
INSTANCE = "a800-57563"
# "base" evaluates the pinned base model itself, with no adapter and no merge.
CHECKPOINTS["base"] = None
E9_ADAPTER = MERGED_MODEL = RUN = None
E6_RUNS = base.HERE / "runs"
VALIDATION_TASKS = ["tomlkit-toml-table-converters", "valibot-recursive-schema-composition",
                    "optique-conditional-option-dependencies"]
TEST_TASKS = [
    "abs-stepped-slices", "cliffy-config-file-parsing", "fastapi-implicit-head-options",
    "gql-incremental-graphql-delivery", "meriyah-explicit-resource-declarations", "numba-stencil-boundary-modes",
    "sqlite-utils-safe-import-checkpoints", "tengo-callable-instance-isolation", "ts-pattern-match-each",
    "sql-formatter-bigquery-pipe-formatting", "mobly-grouped-test-barriers", "scriggo-method-declarations",
    "yjs-map-conflict-detection", "fastapi-deprecation-response-headers",
]
# The E6 held-out split, plus 3 unused passed train60 tasks (prepare_heldout20.py).
ADDED_TEST_TASKS = ["abs-module-cache-flags", "go-genai-streamed-function-args",
                    "tengo-destructuring-bindings"]
HELDOUT_SOURCE = E6_RUNS / "e6-heldout17-source"
SPLIT = "train13"


def heldout_tasks(split: str) -> list[str]:
    """The E6 held-out tasks, optionally with the 3 added test tasks."""

    e6 = json.loads((base.HERE / "data/e6_normal13_loopheavy_v1/split_manifest.json").read_text())
    if e6["validation"] != VALIDATION_TASKS or e6["test"] != TEST_TASKS:
        raise ValueError("Held-out task lists differ from the E6 split manifest")
    if split == "heldout17":
        return VALIDATION_TASKS + TEST_TASKS
    added = json.loads((base.HERE / "data/heldout20_selection_v1/selection.json").read_text())
    if added["validation"] != VALIDATION_TASKS or added["test"] != TEST_TASKS + ADDED_TEST_TASKS:
        raise ValueError("Held-out 20 split differs from its selection manifest")
    return VALIDATION_TASKS + TEST_TASKS + ADDED_TEST_TASKS


def heldout_source(split: str = "heldout17") -> Path:
    """Combine the train13 run configs with the task inputs of the held-out runs."""

    tasks = heldout_tasks(split)
    files = {name: base.SOURCE / "inputs" / name for name in (
        "inference_config.json", "base-stage_model_config.json", "base-controller_config.json", "system-prompt.txt")}
    files |= {f"tasks/{task}.json": E6_RUNS / "e6-validation-20261010-r1/inputs/tasks" / f"{task}.json"
              for task in VALIDATION_TASKS}
    files |= {f"tasks/{task}.json": E6_RUNS / "e6-test-20261010-r1/inputs/tasks" / f"{task}.json"
              for task in TEST_TASKS}
    files |= {f"tasks/{task}.json": base.HERE / "data/heldout20_selection_v1/tasks" / f"{task}.json"
              for task in tasks if task in ADDED_TEST_TASKS}
    for name, source in files.items():
        target = HELDOUT_SOURCE / "inputs" / name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"Existing held-out input differs: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return HELDOUT_SOURCE


def configure(checkpoint: str, run_name: str, instance: str = "a800-57563") -> None:
    global E9_ADAPTER, E9_MATERIALS, MERGED_MODEL, RUN, RAW_BASE, MERGE_PYTHON, MERGE_ROOT, MERGE_SCRIPT, INSTANCE
    global MERGE_PREIMPORT
    E9_ADAPTER = CHECKPOINTS[checkpoint]
    if checkpoint.startswith("e9_1-"):
        E9_MATERIALS = E9_1_MATERIALS
    elif checkpoint.startswith("e10-"):
        E9_MATERIALS = E10_MATERIALS
    INSTANCE, profile = instance, INSTANCES[instance]
    base.SSH_PORT, base.VLLM, base.CACHE_ROOT = profile["port"], profile["vllm"], profile["cache"]
    RAW_BASE, MERGE_PYTHON, MERGE_PREIMPORT = profile["base"], profile["merge_python"], profile.get("merge_preimport")
    MERGE_ROOT = f"{profile['work']}/inference/merge"
    MERGE_SCRIPT = f"{profile['work']}/inference/merge_for_inference.py"
    RUN = base.HERE / "runs" / run_name
    MERGED_MODEL = None if E9_ADAPTER is None else f"/dev/shm/{run_name}-merged"
    base.RUN = RUN
    base.MODEL = "qwen35-9b-e9" if checkpoint == "epoch-1" else f"qwen35-9b-{checkpoint}"
    base.REMOTE_ROOT = f"/root/autodl-tmp/{run_name}"
    base.REMOTE_MODEL = RAW_BASE if E9_ADAPTER is None else MERGED_MODEL
    base.SOCKET = Path(f"/tmp/{run_name}.sock")

_make_inputs = base.make_inputs
_start_remote_service = base.start_remote_service


def make_inputs() -> dict:
    state = _make_inputs()
    prompt = (base.SOURCE / "inputs/system-prompt.txt").read_bytes()
    (RUN / "inputs/system-prompt.txt").write_bytes(prompt)
    config_path = RUN / "inputs/stage_model_config.json"
    config = json.loads(config_path.read_text())
    config["default"]["model_metadata"]["name"] = (
        "Qwen3.5-9B base (no LoRA)" if E9_ADAPTER is None
        else f"Qwen3.5-9B LoRA {CHECKPOINT} (visible teacher notes)")
    base.write_json(config_path, config)
    state.update(
        experiment=f"A-only {SPLIT} evaluation ({CHECKPOINT})", instance=INSTANCE,
        remote_software={"vllm": base.VLLM, "merge_python": MERGE_PYTHON, "merge_preimport": MERGE_PREIMPORT},
        split=SPLIT, task_input_source=str(base.SOURCE),
        budget_control=base.BUDGET if base.BUDGET_ENABLED else None,
        prompt_sha256=hashlib.sha256(prompt).hexdigest(),
        prompt_delta="none; baseline E6/E7 system prompt without the E8 repeat guard",
        model_source={
            "base_model": RAW_BASE, "adapter": E9_ADAPTER, "merged_model": MERGED_MODEL,
            "adapter_training": "none; pinned base model" if E9_ADAPTER is None else
                                "Original base + new LoRA on "
                                + ("E10 (60 rows: E9_1 13 + 47 train60, notes + closing notes)" if CHECKPOINT.startswith("e10-")
                                   else "E9_1 (E9 + closing notes before the final Ledger)" if CHECKPOINT.startswith("e9_1-")
                                   else "E9 (E6 13 normal rows + visible teacher notes)"),
            "materials": E9_MATERIALS,
            "merge_script_sha256": None if E9_ADAPTER is None
                                   else hashlib.sha256(LOCAL_MERGE_SCRIPT.read_bytes()).hexdigest(),
        },
    )
    base.write_json(RUN / "batch.json", state)
    return state


def start_remote_service() -> None:
    if E9_ADAPTER is None:
        _start_remote_service()  # the pinned base model is served directly
        return
    base.remote(f"rm -rf {shlex.quote(MERGE_ROOT)} {shlex.quote(MERGED_MODEL)} "
                f"&& mkdir -p {shlex.quote(MERGE_ROOT)}", timeout=60)
    base.upload(MERGE_SCRIPT, LOCAL_MERGE_SCRIPT.read_bytes())
    # The pinned merge script runs unchanged; an optional pre-import runs before it in the same process.
    preimport = (f"import {MERGE_PREIMPORT}, runpy, sys; sys.argv = sys.argv[1:]; "
                 'runpy.run_path(sys.argv[0], run_name="__main__")')
    launcher = f"-c {shlex.quote(preimport)} " if MERGE_PREIMPORT else ""
    base.remote(
        f"HF_HUB_OFFLINE=1 {shlex.quote(MERGE_PYTHON)} -u {launcher}{shlex.quote(MERGE_SCRIPT)} "
        f"--root {shlex.quote(MERGE_ROOT)} --base {shlex.quote(RAW_BASE)} "
        f"--adapter {shlex.quote(E9_ADAPTER)} --materials {shlex.quote(E9_MATERIALS)} "
        f"--output {shlex.quote(MERGED_MODEL)}",
        timeout=1800,
    )
    _start_remote_service()


def remote(command: str, timeout: int = 120) -> str:
    # The E8 driver stops the service with this call in its cleanup; also free
    # /dev/shm then, since each merged model takes about 18 GB of the 60 GB.
    output = _remote(command, timeout)
    if "server.pid" in command and "kill" in command and MERGED_MODEL:
        _remote(f"sleep 5; rm -rf {shlex.quote(MERGED_MODEL)}", timeout=60)
    return output


_remote = base.remote
base.remote = remote
base.make_inputs = make_inputs
base.start_remote_service = start_remote_service
_summarize = base.summarize


def summarize(state: dict) -> None:
    _summarize(state)
    rows = json.loads((RUN / "comparison.json").read_text()) if (RUN / "comparison.json").exists() else []
    for row in rows:
        manifest = json.loads((RUN / "tasks" / row["task_id"] / "manifest.json").read_text())
        row.update(ledger_quality=manifest.get("ledger_quality"), counts=manifest.get("counts") or {},
                   split="validation" if row["task_id"] in VALIDATION_TASKS
                   else "test" if row["task_id"] in TEST_TASKS + ADDED_TEST_TASKS else "train")
    base.write_json(RUN / "comparison.json", rows)
    lines = [f"# {SPLIT} A-only evaluation ({CHECKPOINT})", "",
             "| Split | Task | State | Quality | O/P/Q | Loop | Model calls | Tool calls | Minutes |",
             "|---|---|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        counts = row["counts"]
        lines.append(
            f"| {row['split']} | {row['task_id']} | {row['final_state']} | {row['ledger_quality'] or '-'} | "
            f"{counts.get('obligations', '-')}/{counts.get('preservation_obligations', '-')}/"
            f"{counts.get('open_questions', '-')} | {'yes' if row['loop_detected'] else 'no'} | "
            f"{row['model_calls']} | {row['tool_calls']}/{row['tool_errors']} | {row['elapsed_sec'] / 60:.1f} |")
    lines.append("")
    for split in dict.fromkeys(row["split"] for row in rows):
        subset = [row for row in rows if row["split"] == split]
        lines.append(f"{split}: completed {sum(row['final_state'] == 'A_COMPLETED' for row in subset)}/{len(subset)}; "
                     f"loop detected {sum(row['loop_detected'] for row in subset)}.")
    (RUN / "comparison.md").write_text("\n".join(lines) + "\n")


base.summarize = summarize

CHECKPOINT = "epoch-1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="epoch-1")
    parser.add_argument("--run-name", default="e9-train13-20261013-r1")
    parser.add_argument("--split", choices=("train13", "heldout17", "heldout20"), default="train13")
    parser.add_argument("--instance", choices=sorted(INSTANCES), default="a800-57563")
    parser.add_argument("--budget-control", action="store_true",
                        help="Enable the inference-time budget gates (off by default)")
    args = parser.parse_args()
    CHECKPOINT, SPLIT = args.checkpoint, args.split
    base.BUDGET_ENABLED = args.budget_control
    if SPLIT.startswith("heldout"):
        base.SOURCE = heldout_source(SPLIT)
        base.TASKS = heldout_tasks(SPLIT)
    configure(args.checkpoint, args.run_name, args.instance)
    base.main()
