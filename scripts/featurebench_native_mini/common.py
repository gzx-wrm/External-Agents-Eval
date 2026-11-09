"""Small, dependency-free helpers for the native FeatureBench experiment."""
from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if secret := os.environ.get("DEEPSEEK_API_KEY"):
        text = text.replace(secret, "[REDACTED]")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n")
    temporary.replace(path)


def cpu_reference_task(task):
    """Keep task semantics intact; Lv2 uses a separate pristine test probe."""
    return copy.deepcopy(task)


def offline_agent_info(output, diagnostics):
    """Use the real saved Agent result, not the probe's diagnostic summary."""
    info = json.loads((output / "native.traj.json").read_text())["info"]
    if info["exit_status"] != "Submitted" or diagnostics["model_api_calls"] != 0:
        raise ValueError("Native offline probe did not complete without model calls")
    return {**diagnostics, "exit_status": info["exit_status"]}


def positive_reference_passed(summary):
    if summary.get("cpu_reference_kind") == "pristine_original_tests":
        return bool(summary.get("completed") and not summary.get("error")
                    and summary.get("probe_passed")
                    and summary.get("tests_executed", 0) > 0
                    and summary.get("test_exit_code") == 0
                    and summary.get("tests_failed") == 0
                    and summary.get("tests_errors") == 0)
    return bool(summary.get("completed") and not summary.get("error")
                and summary.get("resolved") and summary.get("f2p_total", 0) > 0)


def classify_reference_failure(log_text):
    patterns = (
        r"no (?:available )?cuda (?:gpus|devices)", r"no nvidia driver",
        r"found no nvidia driver", r"not compiled with cuda",
        r"cuda is not available", r"requires? (?:an? )?cuda (?:device|gpu)",
        r"cannot find.*(?:libcuda|cuda driver)", r"0 active drivers",
        r"cuda driver.*(?:not found|cannot be found)",
    )
    return "gpu_deferred" if any(re.search(p, log_text, re.IGNORECASE) for p in patterns) else "cpu_validation_failed"
