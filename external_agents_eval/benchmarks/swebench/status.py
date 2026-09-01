from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def collect_status(agent_dir: Path) -> dict[str, Any]:
    manifest_path = agent_dir / "task_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Result manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    task_ids = manifest["task_ids"]
    counters: Counter[str] = Counter()
    details: dict[str, str] = {}
    for instance_id in task_ids:
        path = agent_dir / "predictions" / f"{instance_id}.json"
        if not path.is_file():
            counters["pending"] += 1
            details[instance_id] = "pending"
            continue
        try:
            prediction = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            counters["corrupt"] += 1
            details[instance_id] = "corrupt"
            continue
        status = prediction.get("generation_status", "unknown")
        if status == "success":
            if prediction.get("model_patch", "").strip():
                counters["success_nonempty"] += 1
                details[instance_id] = "success_nonempty"
            else:
                counters["success_empty"] += 1
                details[instance_id] = "success_empty"
        else:
            kind = prediction.get("failure_kind", "failed_unknown")
            label = f"failed_{kind}"
            counters[label] += 1
            details[instance_id] = label
    return {
        "task_count": len(task_ids),
        "counts": dict(sorted(counters.items())),
        "details": details,
    }
