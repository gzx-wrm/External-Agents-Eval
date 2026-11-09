"""Append a frozen, pre-Solver FT ledger to an otherwise unchanged Mini task."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GUIDANCE = (
    "以下清单根据任务描述和基线代码整理，供实现与验证时参考。"
    "请结合原始任务和实际代码判断其适用性；发现遗漏或不一致时自主处理。"
    "交付前检查相关功能与既有行为。"
)
HEADING = "## Frozen FT requirement ledger"


def canonical_sha256(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_frozen_ledger(root: Path, record: dict):
    data = (root / record["ledger_file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["ledger_file_sha256"]:
        raise ValueError("Frozen ledger file changed: " + record["task_id"])
    ledger = json.loads(data)
    if canonical_sha256(ledger) != record["ledger_sha256"]:
        raise ValueError("Frozen ledger content changed: " + record["task_id"])
    return ledger


def augment_task(task: dict, ledger: dict):
    """No evidence packet, review output, or additional agent stage is inserted."""
    text = json.dumps(ledger, ensure_ascii=False, indent=2)
    result = dict(task)
    result["instruction"] = (
        task["instruction"] + "\n\n" + HEADING + "\n\n" + GUIDANCE
        + "\n\n```json\n" + text + "\n```\n"
    )
    return result
