"""Exercise V6.2 wiring in a real isolated image with stub responses, no API calls."""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from run import prepare_repository

import docker
from external_agents_eval.agents.mini_v62 import MiniV62Agent
from external_agents_eval.benchmarks.swebench.isolated_environment import (
    IsolatedEnvironment,
    task_container,
)


class OfflineModel:
    def __init__(self, **kwargs):
        self.calls = 0
        self.identity = uuid.uuid4().hex

    def query(self, messages, **kwargs):
        self.calls += 1
        task = messages[1]["content"]
        call_id = self.identity + "-" + str(self.calls)
        if '"SpecOutput"' in task and self.calls == 1:
            arguments = {"path": "setup.py", "offset": 1, "limit": 5}
            return {"role": "assistant", "content": "Inspect baseline source.",
                    "tool_calls": [{"id": call_id, "type": "function", "function": {
                        "name": "read", "arguments": json.dumps(arguments)}}],
                    "extra": {"actions": [{"name": "read", "arguments": arguments,
                                            "tool_call_id": call_id}], "cost": 0}}
        elif "Before this coding run" in task and self.calls == 1:
            command = "python -c 'import astropy; print(astropy.__file__)'"
        else:
            if "repair_requests" in task:
                answer = {"summary": "Offline controller check.", "decisions": [],
                          "repair_requests": [], "new_questions": []}
            elif "revisit_target_ids" in task:
                answer = {"new_targets": [], "assessments": [], "revisit_target_ids": [], "done": True}
            else:
                answer = {}
            if '"SpecOutput"' in task:
                final = json.dumps(answer)
                return {"role": "assistant", "content": final,
                        "extra": {"actions": [{"final": final}], "cost": 0}}
            command = "submit_json " + shlex.quote(json.dumps(answer))
        return {"role": "assistant", "content": "Offline test response.",
                "tool_calls": [{"id": call_id, "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": command})}}],
                "extra": {"actions": [{"command": command, "tool_call_id": call_id}], "cost": 0}}

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, *args):
        from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

        return format_toolcall_observation_messages(
            actions=message["extra"]["actions"], outputs=outputs,
            observation_template="<returncode>{{output.returncode}}</returncode>\n<output>{{output.output}}</output>",
        )

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads((args.root / "manifests/public_tasks.jsonl").read_text().splitlines()[0])
    image = json.loads((args.root / "preflight" / task["instance_id"] / "image.json").read_text())["id"]
    check_root = args.root / "preflight/offline-pipeline"
    output = check_root / uuid.uuid4().hex[:8]
    client = docker.from_env()
    container = task_container(client, image, name="sv62-offline-" + uuid.uuid4().hex[:8])
    try:
        container.start()
        environment = IsolatedEnvironment(container, instance_id=task["instance_id"])
        asyncio.run(prepare_repository(environment, task))
        agent = MiniV62Agent(
            logs_dir=output / "agent/logs", model_name="deepseek-flash",
            stage_model_config=args.root / "project/configs/v6.1-deepseek-v4.1-flash.json",
            extra_env={"DEEPSEEK_API_KEY": "offline-check-not-a-real-key"},
        )
        ledger = {"summary": "Offline wiring check.", "obligations": [],
                  "preservation_obligations": [], "open_questions": []}
        context = SimpleNamespace(metadata={})
        with patch("minisweagent.models.litellm_model.LitellmModel", OfflineModel), patch(
            "external_agents_eval.workflows.solve_review_repair.ft_inference.FTLedgerClient.generate",
            AsyncMock(return_value=(ledger, {"errors": []})),
        ):
            asyncio.run(agent.run("Offline wiring check.", environment, context))
        manifest = json.loads((output / "agent/v6_review_repair/manifest.json").read_text())
        packet = json.loads((output / "agent/v6_review_repair/v6.2/evidence.json").read_text())
        if manifest["workflow_state"] != "completed" or manifest["route"] != "accept" or not packet["evidence"]:
            raise RuntimeError("Offline workflow did not complete the expected route: " + json.dumps(manifest))
        if packet["tool_counts"] != {"read": 1} or "bash_evidence_audit" in packet:
            raise RuntimeError("A evidence did not use the historical structured-read path")
        solver = json.loads((output / "agent/v6_review_repair/v6/solver/trajectory.json").read_text())
        observations = [item["content"] for step in solver["steps"]
                        for item in step.get("observation", {}).get("results", [])]
        if not any("/tmp/external_eval/worktrees/solver/astropy/__init__.py" in text for text in observations):
            raise RuntimeError("Compiled Astropy runtime was not importable from the Solver worktree")
        report = {"status": "passed", "model_api_calls": 0, "workflow": "v6.2", "framework": "mini-swe-agent",
                  "route": manifest["route"], "evidence_spans": len(packet["evidence"]),
                  "run_directory": str(output),
                  "runtime_import_from_solver_worktree": True,
                  "note": "Stub outputs verify wiring only; this is not a benchmark solution or model smoke result."}
        (output / "result.json").write_text(json.dumps(report, indent=2))
        (check_root / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    finally:
        container.remove(force=True)
        client.close()


if __name__ == "__main__":
    main()
