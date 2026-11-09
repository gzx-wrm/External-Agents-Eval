"""Run E8_1 on the E7 post-trained Qwen3.5-9B checkpoint."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from a_agent_finetune import run_e8_base_train13 as base


RUN = base.HERE / "runs/e8_1-e7-train13-20261012-r1"
MERGE_ROOT = "/root/autodl-tmp/e7/inference/e8_1-merge"
MERGE_SCRIPT = "/root/autodl-tmp/e7/inference/e8_1-merge.py"
E7_ADAPTER = "/root/autodl-tmp/e7/runs/e7-normal13-recovery4-20261012-r2/checkpoint-epoch-1/adapter"
E7_MATERIALS = "/root/autodl-tmp/e7/e7_normal13_recovery4_v1"
MERGED_MODEL = "/dev/shm/e8_1-e7-20261012-r1-merged"
RAW_BASE = "/root/autodl-tmp/g1/models/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

base.RUN = RUN
base.MODEL = "qwen35-9b-e7-e8_1"
base.REMOTE_ROOT = "/root/autodl-tmp/e8_1-e7-train13-20261012-r1"
base.REMOTE_MODEL = MERGED_MODEL
base.SOCKET = Path("/tmp/e8_1-e7-train13-20261012.sock")


_make_inputs = base.make_inputs


def make_inputs() -> dict[str, str]:
    state = _make_inputs()
    config_path = RUN / "inputs/stage_model_config.json"
    config = json.loads(config_path.read_text())
    config["default"]["model_metadata"]["name"] = (
        "Qwen3.5-9B E7 post-trained model with E8 repeat guard"
    )
    base.write_json(config_path, config)
    state["experiment"] = "E8_1 E7 post-trained Qwen3.5-9B with explicit tool-repeat guard"
    state["model_source"] = {
        "base_model": base.REMOTE_MODEL,
        "adapter": E7_ADAPTER,
        "merged_model": MERGED_MODEL,
        "adapter_training": "E6 epoch-1 continuation + E7 recovery SFT",
    }
    base.write_json(RUN / "batch.json", state)
    return state


def start_remote_service() -> None:
    merge_path = Path(base.HERE / "data/e7_normal13_recovery4_v1/provenance/merge_e7_for_inference.py")
    base.remote(
        f"rm -rf {shlex.quote(MERGE_ROOT)} {shlex.quote(MERGED_MODEL)} "
        f"&& mkdir -p {shlex.quote(MERGE_ROOT)}",
        timeout=60,
    )
    base.upload(MERGE_SCRIPT, merge_path.read_bytes())
    base.remote(f"chmod 644 {shlex.quote(MERGE_SCRIPT)}", timeout=30)
    merge_command = (
        f"HF_HUB_OFFLINE=1 /root/autodl-tmp/g1/venv/bin/python -u "
        f"{shlex.quote(MERGE_SCRIPT)} "
        f"--root {shlex.quote(MERGE_ROOT)} "
        f"--base {shlex.quote(RAW_BASE)} "
        f"--adapter {shlex.quote(E7_ADAPTER)} "
        f"--materials {shlex.quote(E7_MATERIALS)} "
        f"--output {shlex.quote(MERGED_MODEL)}"
    )
    # base.REMOTE_MODEL is the merged output for service; merge from the pinned raw base.
    base.remote(merge_command, timeout=1800)
    _original_start()


_original_start = base.start_remote_service
base.start_remote_service = start_remote_service


if __name__ == "__main__":
    base.main()
