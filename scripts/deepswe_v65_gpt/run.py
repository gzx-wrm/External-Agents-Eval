"""V6.5 with a cross-model review: GPT-6.1 Sol for X/D/I/R/S, DeepSeek Solver and Repair.

This launcher reuses scripts/deepswe_v65/run.py unchanged and replaces only the review
agent (MiniV65GptReviewAgent), the experiment IDs and the frozen-source list. GPT
endpoint credentials stay in the existing host-private file named by
GPT_REVIEW_ENDPOINTS_FILE; DeepSeek (Repair) still needs DEEPSEEK_API_KEY.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.agents.mini_v65_gpt import (  # noqa: E402
    MiniV65GptReviewAgent,
    V65GptConfig,
    initial_endpoint_for,
)

DEFAULT_ENDPOINTS = "/root/.config/external-agents-eval/ft-m1-gpt61sol-20261030/endpoints.json"


def _load_base():
    spec = importlib.util.spec_from_file_location("v65_base_runner", PROJECT / "scripts/deepswe_v65/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.__doc__ = __doc__
base.EXPERIMENT_ID = "v6.5-gpt-review"
base.PILOT_EXPERIMENT_ID = "v6.5-gpt-review-pilot"
base.FROZEN_SOURCES = [
    *base.FROZEN_SOURCES,
    "scripts/deepswe_v65_gpt/run.py",
    "external_agents_eval/agents/mini_v65_gpt.py",
    "scripts/deepswe_ft_m1_gpt/routed_model.py",
]
base.V65Config = V65GptConfig  # recorded in selection.json as review_config


def endpoints_file() -> Path:
    return Path(os.environ.get("GPT_REVIEW_ENDPOINTS_FILE", DEFAULT_ENDPOINTS))


def review_agent(**kwargs):
    # logs_dir = <root>/outputs/<task>/review/agent/logs
    task_id = Path(kwargs["logs_dir"]).parents[2].name
    return MiniV65GptReviewAgent(endpoints_file=endpoints_file(),
                                 initial_endpoint=initial_endpoint_for(task_id), **kwargs)


base.MiniV65ReviewAgent = review_agent
_save = base.save


def save(path, value):
    _save(path, value)
    target = Path(path)
    if endpoints_file().is_file() and target.is_file():
        text = target.read_text()
        for endpoint in json.loads(endpoints_file().read_text()):
            text = text.replace(endpoint["api_key"], "[REDACTED]")
        target.write_text(text)


base.save = save
_run_batch = base.run_batch


def run_batch(*args, **kwargs):
    if not endpoints_file().is_file():
        raise ValueError("Supply the existing host-private GPT endpoints file")
    return _run_batch(*args, **kwargs)


base.run_batch = run_batch
main = base.main

if __name__ == "__main__":
    main()
