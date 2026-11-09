"""V6.6 transport retry: resend only when no valid response arrived."""
from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from external_agents_eval.workflows.solve_review_repair import ft_inference
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore, sha256_json
from external_agents_eval.workflows.solve_review_repair.ft_inference import FTLedgerClient
from external_agents_eval.workflows.solve_review_repair.v66_checkpoints import CheckpointClient
from external_agents_eval.workflows.solve_review_repair.v66_transport import (
    RETRY_WAITS_SEC,
    NoValidResponse,
    V66FTLedgerClient,
    open_with_retry,
)

GOOD = json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}).encode()


class Reply:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return HTTPError("http://x", code, "err", {}, io.BytesIO(b'{"error": "busy"}'))


def opener_from(outcomes, calls):
    def opener(request, timeout):
        calls.append(timeout)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return Reply(outcome)
    return opener


@pytest.mark.parametrize("failure", [
    http_error(503), http_error(429), http_error(500), URLError("reset"), TimeoutError("slow"),
    b"", b"\n\n\nnull", b"<html>busy</html>", json.dumps({"error": {"message": "busy"}}).encode(),
])
def test_retries_only_until_a_response_with_choices(failure):
    calls, sleeps, records = [], [], []
    response = open_with_retry("req", timeout=7, record=records.append, sleep=sleeps.append,
                               opener=opener_from([failure, GOOD], calls))
    assert response.read() == GOOD and response.status == 200
    assert calls == [7, 7] and sleeps == [RETRY_WAITS_SEC[0]]
    assert [r["outcome"] for r in records] == [
        "http_error" if isinstance(failure, HTTPError) else "no_valid_response", "response"]


def test_delivered_output_is_never_resent_even_if_unusable():
    calls = []
    unusable = json.dumps({"choices": [{"message": {"content": "not the schema"}}]}).encode()
    response = open_with_retry("req", timeout=1, record=lambda e: None, sleep=lambda s: None,
                               opener=opener_from([unusable], calls))
    assert response.read() == unusable and len(calls) == 1


@pytest.mark.parametrize("code", [400, 401, 402, 404])
def test_client_errors_are_not_retried(code):
    calls = []
    with pytest.raises(HTTPError):
        open_with_retry("req", timeout=1, record=lambda e: None, sleep=lambda s: None,
                        opener=opener_from([http_error(code), GOOD], calls))
    assert len(calls) == 1


def test_gives_up_after_three_retries_with_backoff():
    calls, sleeps, records = [], [], []
    with pytest.raises(NoValidResponse):
        open_with_retry("req", timeout=1, record=records.append, sleep=sleeps.append,
                        opener=opener_from([b"null"] * 4, calls))
    assert len(calls) == 4 and sleeps == list(RETRY_WAITS_SEC) and len(records) == 4


def test_v66_ft_client_runs_the_frozen_ft_code_with_only_the_transport_replaced():
    client = V66FTLedgerClient(model="m", base_url="http://x", api_key="k")
    assert client._frozen_generate.__code__ is FTLedgerClient._generate.__code__
    globals_ = client._frozen_generate.__globals__
    differing = {name for name, value in vars(ft_inference).items()
                 if name in globals_ and globals_[name] is not value}
    assert differing == {"urlopen"} and globals_["urlopen"] == client._open
    assert ft_inference.urlopen.__module__ == "urllib.request"  # the frozen module is untouched


def ft_sample():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    return {"task_id": "t", "base_commit": "b", "messages": messages,
            "messages_sha256": sha256_json(messages)}


def test_v66_ft_client_retries_a_null_body_then_records_the_transport(tmp_path, monkeypatch):
    calls, sleeps = [], []
    output = {"summary": "s", "obligations": [], "preservation_obligations": [], "open_questions": []}
    body = json.dumps({"choices": [{"message": {"content": json.dumps(output), "reasoning_content": "r"},
                                    "finish_reason": "stop"}], "usage": {}}).encode()
    client = V66FTLedgerClient(model="m", base_url="http://x", api_key="k", sleep=sleeps.append,
                               opener=opener_from([b"\n\nnull", body], calls))
    monkeypatch.setattr(ft_inference, "validate_output", lambda text, sample, validator: (
        output, {"schema_valid": True, "sources_valid": True, "errors": []}))
    client._frozen_generate.__globals__["validate_output"] = ft_inference.validate_output
    store = ArtifactStore(tmp_path)
    payload, checks = client._generate(ft_sample(), store)
    assert payload == output and len(calls) == 2 and sleeps == [RETRY_WAITS_SEC[0]]
    transport = json.loads((tmp_path / "v6.2/ledger/transport.json").read_text())
    assert [entry["outcome"] for entry in transport] == ["no_valid_response", "response"]
    result = json.loads((tmp_path / "v6.2/ledger/result.json").read_text())
    assert result["status"] == "passed" and result["attempts"] == 1  # one FT sample


def test_checkpoint_client_uses_the_retrying_transport(tmp_path):
    calls, sleeps = [], []
    client = CheckpointClient(model="m", base_url="http://x", api_key="k", max_per_target=4,
                              max_attempts=1, timeout_sec=3, sleep=sleeps.append,
                              opener=opener_from([http_error(503), GOOD], calls))
    assert client._post([{"role": "user", "content": "u"}])["choices"]
    assert calls == [3, 3] and [e["outcome"] for e in client.transport] == ["http_error", "response"]
