"""Frozen FT ledger + native Mini, changing only the downstream model transport."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import m0_run as native
import yaml
from ledger_input import augment_task, load_frozen_ledger


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, sources, review_root):
    if root.exists():
        raise FileExistsError("Use a new experiment directory")
    review_file = review_root / "result.json"
    review = json.loads(review_file.read_text())
    failed = {x["task_id"] for x in review["results"] if x.get("reward") == 0}
    if len(failed) != 5:
        raise ValueError("Expected the five final ft_m1_review failures")
    records = {}
    for source in sources:
        selection = json.loads((source / "selection.json").read_text())
        original = {x["task_id"]: x for x in native.load_tasks(source)}
        for record in selection["tasks"]:
            name = record["task_id"]
            if name not in failed:
                continue
            ledger = load_frozen_ledger(source, record)
            task = augment_task(original[name], ledger)
            task["image"] = record["m0_image"]["repo_digests"][0]
            previous = source / "inputs" / name / "task.json"
            if task != json.loads(previous.read_text()):
                raise ValueError("The proposed model input differs from M1: " + name)
            records[name] = (source, record, task)
    if records.keys() != failed:
        raise ValueError("Not all five frozen M1 inputs were found")
    root.mkdir(parents=True)
    (root / "scripts").mkdir()
    for name in ["run.py", "launch.py", "routed_model.py", "m0_run.py", "ledger_input.py", "sanitize_repository.py"]:
        shutil.copyfile(Path(__file__).parent / name, root / "scripts" / name)
    for name in ["grader", "manifests"]:
        (root / name).symlink_to(sources[0] / name, target_is_directory=True)
    (root / "configs").mkdir()
    original_config = sources[0] / "configs/native-deepseek-v4.1-flash.yaml"
    shutil.copyfile(original_config, root / "configs/frozen-m1.yaml")
    chosen = []
    for task in native.load_tasks(root):
        name = task["task_id"]
        if name not in records:
            continue
        source, record, augmented = records[name]
        if yaml.safe_load((source / "configs/native-deepseek-v4.1-flash.yaml").read_text()) != yaml.safe_load(original_config.read_text()):
            raise ValueError("Source M1 configurations differ")
        ledger_file = root / "inputs" / name / "ledger.json"
        ledger_file.parent.mkdir(parents=True)
        shutil.copyfile(source / record["ledger_file"], ledger_file)
        task_file = ledger_file.parent / "task.json"
        native.save(task_file, augmented)
        chosen.append({"task_id": name, "task_checksum": task["task_checksum"],
                       "task_file": str(task_file.relative_to(root)), "task_file_sha256": digest(task_file),
                       "ledger_file": str(ledger_file.relative_to(root)),
                       "ledger_sha256": record["ledger_sha256"],
                       "ledger_file_sha256": digest(ledger_file), "image": record["m0_image"],
                       "m1_root": str(source), "m1_result_sha256": digest(source / "outputs" / name / "result.json"),
                       "original_instruction_sha256": hashlib.sha256(task["instruction"].encode()).hexdigest(),
                       "augmented_instruction_sha256": hashlib.sha256(augmented["instruction"].encode()).hexdigest()})
    native.save(root / "selection.json", {
        "experiment_id": "ft_m1_gpt61sol", "tasks": chosen, "task_count": len(chosen),
        "review_failure_source": str(review_file), "review_failure_source_sha256": digest(review_file),
        "fresh_solver_from_original_repository": True, "previous_patch_or_conversation_reused": False,
        "evidence_packet_included": False, "review_guidance_included": False,
        "model": "gpt-6.1-sol", "reasoning_effort": None, "settings_approval_pending": True,
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in [
            *sorted((root / "scripts").glob("*.py")), root / "configs/frozen-m1.yaml"]},
        "official_gold_control_reference": str(sources[0] / "preflight/readiness.json"),
    })
    print(json.dumps({"prepared": True, "task_count": len(chosen), "root": str(root)}), flush=True)


def configuration(root, effort, endpoints_file):
    config = yaml.safe_load((root / "configs/frozen-m1.yaml").read_text())
    config = copy.deepcopy(config)
    config["model"].update(model_name="openai/gpt-6.1-sol", model_class="routed_model.RoutedResponseModel",
                           endpoints_file=str(endpoints_file), route_log=str(root / "preflight/routes.jsonl"),
                           initial_endpoint=0)
    config["model"]["model_kwargs"] = {
        "reasoning": {"effort": effort}, "max_output_tokens": 65536,
        "store": False, "include": ["reasoning.encrypted_content"],
    }
    return config


def offline_probe(config, environment, output, task):
    from routed_model import RoutedResponseModel

    received = []

    def request(self, endpoint, *, instructions, messages, parameters):
        received.append(copy.deepcopy(messages))
        if len(received) == 2:
            assert any(m.get("type") == "function_call_output" for m in messages)
            assert any(m.get("type") == "function_call" for m in messages)
            reasoning = next(m for m in messages if m.get("type") == "reasoning")
            assert reasoning["encrypted_content"] == "opaque-probe"
            assert "status" not in reasoning and "content" not in reasoning
        command = ("printf 'native offline probe\\n' > native_offline_probe.txt" if len(received) == 1
                   else "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        data = {"object": "response", "model": "gpt-6.1-sol", "status": "completed", "output": [
            {"type": "reasoning", "id": "rs_probe_" + str(len(received)), "summary": [],
             "encrypted_content": "opaque-probe", "status": "completed", "content": None},
            {"type": "function_call", "id": "fc_probe_" + str(len(received)),
             "call_id": "call_probe_" + str(len(received)), "name": "bash",
             "arguments": json.dumps({"command": command}), "status": "completed"}]}
        return SimpleNamespace(**data, usage=None, model_dump=lambda **_: copy.deepcopy(data))

    agent = native.make_agent(config, environment, output)
    with patch.object(RoutedResponseModel, "_request", request):
        info = agent.run(task["instruction"])
    if info["exit_status"] != "Submitted" or len(received) != 2:
        raise ValueError("Actual native Mini Responses/Bash protocol failed")
    if not any(m.get("content", "").find(task["instruction"]) >= 0
               for m in received[0] if isinstance(m.get("content"), str)):
        raise ValueError("Frozen augmented task did not reach Mini")
    return {"model_api_calls": 0, "responses_bash_roundtrip_verified": True,
            "reasoning_metadata_input_compatibility_verified": True,
            "actual_native_agent": type(agent).__module__ + "." + type(agent).__name__}


def load(root):
    selection = json.loads((root / "selection.json").read_text())
    native.load_tasks(root)  # Verify the same suite and official grader bytes.
    tasks = []
    for relative, expected in selection["source_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    for record in selection["tasks"]:
        for field, checksum in [("task_file", "task_file_sha256"), ("ledger_file", "ledger_file_sha256")]:
            if digest(root / record[field]) != record[checksum]:
                raise ValueError("Frozen input changed")
        tasks.append(json.loads((root / record["task_file"]).read_text()))
    return selection, tasks


def run(root, selection, tasks, config):
    if not json.loads((root / "preflight/readiness.json").read_text())["passed"]:
        raise ValueError("Offline preflight has not passed")
    credentials = json.loads(Path(config["model"]["endpoints_file"]).read_text())
    secrets = [x["api_key"] for x in credentials]
    lock = threading.Lock()
    slots = threading.Condition()
    active_slots = 0
    states = {t["task_id"]: {"phase": "queued"} for t in tasks}

    def parent_active_slots():
        parent = selection.get("shared_concurrency_parent_root")
        if not parent:
            return 0
        status = json.loads((Path(parent) / "monitor-status.json").read_text())
        return sum(value.get("phase") in {"image_preparation", "solving", "official_evaluation"}
                   for value in status["tasks"].values())

    def update(task, phase, **values):
        with lock:
            states[task["task_id"]].update(phase=phase, updated_at=native.now(), **values)
            native.save(root / "monitor-status.json", {"workers": 3, "task_count": len(tasks), "tasks": states})
        print(json.dumps({"task_id": task["task_id"], "phase": phase, **values}), flush=True)

    def work(index, task):
        output = root / "outputs" / task["task_id"]
        output.mkdir(parents=True, exist_ok=True)
        if (output / "result.json").exists():
            result = json.loads((output / "result.json").read_text())
            update(task, result["status"], reused_existing_outcome=True)
            return result
        client, added = native.docker.from_env(timeout=1800), False
        try:
            update(task, "image_preparation")
            image, added = native.get_image(client, task, output)
            if image.id != selection["tasks"][index]["image"]["id"]:
                raise ValueError("Original image ID changed")
            individual = copy.deepcopy(config)
            individual["model"].update(initial_endpoint=index % 2, route_log=str(output / "endpoint-routes.jsonl"))
            update(task, "solving", initial_endpoint=index % 2 + 1, started_at=native.now())
            model_patch, info = native.generate(task, root, output, individual, image, "generate")
            prediction = json.loads((output / "prediction.json").read_text())
            prediction["model"] = "gpt-6.1-sol"
            native.save(output / "prediction.json", prediction)
            update(task, "official_evaluation", exit_status=info.get("exit_status"))
            result = native.grade(root, task, image.id, model_patch, output / "evaluation")
            result.update(exit_status=info.get("exit_status"), model="gpt-6.1-sol",
                          native_agent_completed=info.get("exit_status") == "Submitted",
                          patch_sha256=hashlib.sha256(model_patch.encode()).hexdigest())
        except Exception as error:
            text = str(error)
            for secret in secrets:
                text = text.replace(secret, "[REDACTED]")
            result = {"status": "infrastructure_or_stage_error", "task_id": task["task_id"],
                      "reward": None, "error_type": type(error).__name__, "error": text}
        finally:
            if added:
                try:
                    client.images.remove(task["image"], noprune=False)
                except native.docker.errors.APIError:
                    pass
            client.close()
        result["finished_at"] = native.now()
        native.save(output / "result.json", result)
        update(task, result["status"], reward=result.get("reward"))
        return result

    def one(index, task):
        nonlocal active_slots
        update(task, "waiting_for_slot")
        with slots:
            while active_slots + parent_active_slots() >= 3:
                slots.wait(timeout=2)
            active_slots += 1
        try:
            return work(index, task)
        finally:
            with slots:
                active_slots -= 1
                slots.notify_all()

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(as_completed([pool.submit(one, i, t) for i, t in enumerate(tasks)]))
        results = [x.result() for x in results]
    native.save(root / "result.json", {"task_count": len(tasks), "results": results, "finished_at": native.now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, action="append")
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--reasoning-effort", choices=["high", "xhigh"])
    parser.add_argument("--endpoints-file", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        prepare(root, args.source_run, args.review_root)
        return
    if not args.reasoning_effort or not args.endpoints_file:
        raise ValueError("Explicitly approved reasoning effort and host endpoint file required")
    selection, tasks = load(root)
    config = configuration(root, args.reasoning_effort, args.endpoints_file)
    if args.mode == "preflight":
        output = root / "preflight" / tasks[0]["task_id"]
        client = native.docker.from_env(timeout=1800)
        try:
            image, _ = native.get_image(client, tasks[0], output)
            with patch.object(native, "offline_probe", offline_probe):
                _, protocol = native.generate(tasks[0], root, output / "offline", config, image, "offline")
        finally:
            client.close()
        reference = json.loads(Path(selection["official_gold_control_reference"]).read_text())
        if not reference["passed"]:
            raise ValueError("Original official scorer validation is invalid")
        native.save(root / "preflight/readiness.json", {
            "passed": True, "offline_protocol": protocol, "model_api_calls": 0,
            "official_gold_control_reused_from": selection["official_gold_control_reference"],
            "all_frozen_inputs_verified": True, "reasoning_effort": args.reasoning_effort,
        })
    else:
        run(root, selection, tasks, config)


if __name__ == "__main__":
    main()
