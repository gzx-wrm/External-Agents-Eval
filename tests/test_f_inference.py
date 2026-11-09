from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from jsonschema import Draft202012Validator

from a_agent_finetune import run_f_inference as runner
from external_agents_eval.workflows.solve_review_repair.v5_prompts import SPEC_OUTPUT_SCHEMA

DATA_DIR = runner.HERE / "data" / "f_pilot_5"
SAMPLE = {
    "task": {"title": "Add retries", "description": "Retry transient failures."},
    "evidence": [
        {
            "path": "retry.py",
            "line_start": 10,
            "line_end": 12,
            "text": "def retry():\n    pass\n\n",
        },
    ],
}


def payload() -> dict:
    return {
        "summary": "Add bounded retries.",
        "obligations": [
            {
                "statement": "Retry transient failures.",
                "importance": "required",
                "sources": [
                    {"kind": "task", "field": "description", "quote": "Retry transient failures."}
                ],
            }
        ],
        "preservation_obligations": [
            {
                "statement": "Preserve the retry function.",
                "importance": "required",
                "sources": [
                    {"kind": "repository", "path": "retry.py", "line_start": 10, "line_end": 11}
                ],
            }
        ],
        "open_questions": [],
    }


def validate(value: dict) -> dict:
    _, checks = runner.validate_output(
        json.dumps(value), SAMPLE, Draft202012Validator(SPEC_OUTPUT_SCHEMA)
    )
    return checks


def test_exported_schema_matches_workflow_contract():
    schema = json.loads((runner.HERE / "spec_output_schema.json").read_text())
    assert schema == SPEC_OUTPUT_SCHEMA
    Draft202012Validator.check_schema(schema)
    assert validate(payload())["sources_valid"]


@pytest.mark.parametrize("variant", ["32k", "64k", "full"])
def test_real_samples_bind_exact_prompts_to_records(variant):
    samples, data_file = runner.load_samples(DATA_DIR, variant)
    rows = [json.loads(line) for line in data_file.read_text().splitlines()]
    assert len(samples) == len(rows) == 5
    assert samples[0]["task_id"] == "abs-stepped-slices"
    assert [sample["messages"] for sample in samples] == [row["prompt"] for row in rows]
    assert all("completion" not in sample for sample in samples)


@pytest.mark.parametrize(
    "text",
    [
        "```json\n{}\n```",
        '{"summary": "one", "summary": "two"}',
        '{"summary": NaN}',
    ],
)
def test_invalid_json_is_recorded_without_repair(text):
    output, checks = runner.validate_output(text, SAMPLE, Draft202012Validator(SPEC_OUTPUT_SCHEMA))
    assert output is None
    assert not checks["json_valid"]
    assert checks["errors"][0]["check"] == "json"


@pytest.mark.parametrize("case", ["blank", "duplicate_statement", "reversed", "float", "extra"])
def test_schema_contract_failures(case):
    value = payload()
    source = value["preservation_obligations"][0]["sources"][0]
    if case == "blank":
        value["summary"] = " \n "
    elif case == "duplicate_statement":
        value["preservation_obligations"][0]["statement"] = " RETRY   transient failures. "
    elif case == "reversed":
        source["line_start"], source["line_end"] = 11, 10
    elif case == "float":
        source["line_start"] = 10.0
    elif case == "extra":
        value["obligations"][0]["id"] = "O001"
    checks = validate(value)
    assert checks["json_valid"]
    assert not checks["schema_valid"]
    assert not checks["sources_valid"]


@pytest.mark.parametrize("case", ["quote", "path", "range", "empty"])
def test_sources_must_be_visible_in_input(case):
    value = payload()
    source = value["preservation_obligations"][0]["sources"][0]
    if case == "quote":
        value["obligations"][0]["sources"][0]["quote"] = "Unwritten requirement"
    elif case == "path":
        source["path"] = "private_test.py"
    elif case == "range":
        source["line_start"], source["line_end"] = 9, 11
    elif case == "empty":
        source["line_start"], source["line_end"] = 12, 12
    checks = validate(value)
    assert checks["schema_valid"]
    assert not checks["sources_valid"]
    assert checks["errors"][0]["check"] == "source"


@pytest.mark.parametrize("limit", [None, 1])
def test_batch_records_success_truncation_failure_and_only_sends_prompts(
    tmp_path, monkeypatch, limit
):
    samples, _ = runner.load_samples(DATA_DIR, "32k")
    responses = []
    for index, sample in enumerate(samples):
        value = payload()
        value["preservation_obligations"] = []
        value["obligations"][0]["sources"] = [
            {
                "kind": "task",
                "field": "title",
                "quote": sample["task"]["title"],
            }
        ]
        if index == 3:
            value["obligations"][0]["sources"][0]["quote"] = "Unwritten requirement"
        response = {
            "id": f"response-{index}",
            "model": "Qwen/Qwen3.5-9B",
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(value)},
                    "finish_reason": {1: "length", 4: "content_filter"}.get(index, "stop"),
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        responses.append((503, {"error": "Temporary outage"}) if index == 2 else (200, response))

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                {"path": self.path, "body": body, "authorization": self.headers["Authorization"]}
            )
            status, response = responses[len(requests) - 1]
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("F_INFERENCE_API_KEY", "test-secret-not-for-files")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_dir = tmp_path / "run"
    try:
        args = [
            "--data-dir",
            str(DATA_DIR),
            "--output-dir",
            str(output_dir),
            "--base-url",
            f"http://127.0.0.1:{server.server_port}/v1",
        ]
        if limit is not None:
            args.extend(["--limit", str(limit)])
        code = runner.main(args)
        with pytest.raises(SystemExit) as error:
            runner.main(args)
        assert error.value.code == 2  # An existing run must not be overwritten.
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    planned = limit or len(samples)
    assert code == (0 if limit == 1 else 1)
    assert len(requests) == planned
    for request, sample in zip(requests, samples[:planned], strict=True):
        assert request["path"] == "/v1/chat/completions"
        assert request["body"]["messages"] == sample["messages"]
        assert set(request["body"]) == {
            "model",
            "messages",
            "max_tokens",
            "temperature",
            "seed",
            "stream",
            "chat_template_kwargs",
        }
        assert request["body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert request["authorization"] == "Bearer test-secret-not-for-files"
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["planned_tasks"] == summary["completed_tasks"] == planned
    assert summary["passed"] == 1
    for key in ("truncated", "request_errors", "incomplete", "invalid_outputs"):
        assert summary[key] == (0 if limit == 1 else 1)
    completed_requests = 1 if limit == 1 else 4
    assert summary["usage"] == {
        "prompt_tokens": 10 * completed_requests,
        "completion_tokens": 5 * completed_requests,
        "total_tokens": 15 * completed_requests,
    }
    assert [row["status"] for row in summary["tasks"]] == [
        "passed",
        "truncated",
        "request_error",
        "invalid_output",
        "incomplete",
    ][:planned]
    assert all(row["elapsed_seconds"] >= 0 for row in summary["tasks"])
    assert not summary["semantic_support_checked"]
    for index, sample in enumerate(samples[:planned]):
        task_dir = output_dir / sample["task_id"]
        assert (task_dir / "raw_response.txt").exists()
        assert (task_dir / "request.json").exists()
        assert (task_dir / "result.json").exists()
        if index != 2:
            assert (task_dir / "response.json").exists()
            assert (task_dir / "output.txt").exists()
            assert (task_dir / "spec_output.json").exists()
            assert (task_dir / "validation.json").exists()
    for file in output_dir.rglob("*"):
        if file.is_file():
            assert "test-secret-not-for-files" not in file.read_text()


def test_mismatched_evidence_is_rejected_before_any_request(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "records").mkdir(parents=True)
    (data_dir / "records_32k").mkdir()
    record = copy.deepcopy(SAMPLE)
    record.update(task_id="one", base_commit="baseline", managed_delivery=[])
    runner.write_json(data_dir / "manifest.json", {"tasks": [{"task_id": "one"}]})
    runner.write_json(data_dir / "records" / "one.json", record)
    runner.write_json(data_dir / "records_32k" / "one.json", record)
    row = {
        "prompt": [
            {"role": "system", "content": "Output JSON"},
            {"role": "user", "content": "Different evidence"},
        ],
        "completion": [{"role": "assistant", "content": "Not sent"}],
    }
    (data_dir / "training_candidates_32k.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Prompt does not match"):
        runner.load_samples(data_dir, "32k")


def test_named_task_sends_only_that_tasks_prompt(tmp_path, monkeypatch):
    samples, _ = runner.load_samples(DATA_DIR, "full")
    chosen = samples[-1]
    received = []

    def fake_run_one(sample, args, validator):
        received.append(sample)
        return {
            "task_id": sample["task_id"], "status": "passed", "elapsed_seconds": 0,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    output = tmp_path / "selected"
    args = ["--data-dir", str(DATA_DIR), "--variant", "full", "--output-dir", str(output), "--task-id", chosen["task_id"]]
    assert runner.main(args) == 0
    assert received == [chosen]
    assert json.loads((output / "run.json").read_text())["task_ids"] == [chosen["task_id"]]
    assert json.loads((output / "summary.json").read_text())["planned_tasks"] == 1


@pytest.mark.parametrize("selector", ["--task-id", "--exclude-task-id"])
def test_missing_named_task_is_rejected_without_creating_run(tmp_path, monkeypatch, selector):
    def unexpected_request(*args):
        pytest.fail("No model request should be sent for an unknown task")

    monkeypatch.setattr(runner, "run_one", unexpected_request)
    output = tmp_path / "unknown"
    with pytest.raises(SystemExit) as error:
        runner.main(["--data-dir", str(DATA_DIR), "--output-dir", str(output), selector, "missing-task"])
    assert error.value.code == 2
    assert not output.exists()


def test_excluded_completed_task_is_not_requested(tmp_path, monkeypatch):
    samples, _ = runner.load_samples(DATA_DIR, "full")
    excluded = samples[-1]["task_id"]
    received = []

    def fake_run_one(sample, args, validator):
        received.append(sample)
        return {
            "task_id": sample["task_id"], "status": "passed", "elapsed_seconds": 0,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    output = tmp_path / "remaining"
    assert runner.main([
        "--data-dir", str(DATA_DIR), "--variant", "full", "--output-dir", str(output),
        "--exclude-task-id", excluded,
    ]) == 0
    assert received == samples[:-1]
    assert json.loads((output / "run.json").read_text())["task_ids"] == [s["task_id"] for s in samples[:-1]]
    assert json.loads((output / "summary.json").read_text())["planned_tasks"] == len(samples)-1
