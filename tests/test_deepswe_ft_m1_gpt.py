from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import httpx
    import openai

    importlib.import_module("minisweagent")
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Optional Mini/OpenAI runtime is not installed") from error

MODULE = Path(__file__).resolve().parents[1] / "scripts/deepswe_ft_m1_gpt/routed_model.py"
SPEC = importlib.util.spec_from_file_location("ft_gpt_routed_model", MODULE)
route = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = route
SPEC.loader.exec_module(route)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.credentials = [{"id": "endpoint1", "base_url": "https://one.invalid/v1", "api_key": "test-private-one"},
                            {"id": "endpoint2", "base_url": "https://two.invalid/v1", "api_key": "test-private-two"}]
        path = root / "private.json"
        path.write_text(json.dumps(self.credentials))
        self.model = route.RoutedResponseModel(
            model_name="openai/gpt-6.1-sol", endpoints_file=path,
            route_log=root / "routes.jsonl", cost_tracking="ignore_errors",
            model_kwargs={"reasoning": {"effort": "xhigh"}, "max_output_tokens": 65536},
        )
        self.messages = [{"role": "system", "content": "Original Mini system prompt.\n"},
                         {"role": "user", "content": "Original issue and frozen ledger."},
                         {"type": "reasoning", "id": "rs_old", "encrypted_content": "opaque-state"},
                         {"type": "function_call", "name": "bash", "call_id": "call_old", "arguments": "{\"command\":\"ls\"}"},
                         {"type": "function_call_output", "call_id": "call_old", "output": "file.py"}]
        self.response = SimpleNamespace(model="gpt-6.1-sol", status="completed", usage=None)

    def test_failover_preserves_all_history_and_sticks_to_successful_endpoint(self):
        calls = []

        def request(endpoint, **kwargs):
            calls.append((endpoint["id"], copy.deepcopy(kwargs)))
            if len(calls) == 1:
                raise openai.APIConnectionError(request=httpx.Request("POST", "https://one.invalid/v1/responses"))
            return self.response

        before = copy.deepcopy(self.messages)
        with patch.object(self.model, "_request", request):
            self.model._query(self.messages)
            self.model._query(self.messages)
        self.assertEqual([x[0] for x in calls], ["endpoint1", "endpoint2", "endpoint2"])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][1]["instructions"], before[0]["content"])
        self.assertEqual(calls[0][1]["messages"], before[1:])
        self.assertEqual(self.messages, before)

    def test_bad_request_is_explicit_and_does_not_silently_change_context(self):
        error = openai.BadRequestError("Invalid opaque history", response=httpx.Response(
            400, request=httpx.Request("POST", "https://one.invalid/v1/responses")), body=None)
        with patch.object(self.model, "_request", side_effect=error) as request:
            with self.assertRaises(openai.BadRequestError):
                self.model._query(self.messages)
        self.assertEqual(request.call_count, 1)

    def test_credentials_never_enter_serialized_config_or_route_log(self):
        with patch.object(self.model, "_request", return_value=self.response):
            self.model._query(self.messages)
        text = json.dumps(self.model.serialize()) + self.model.config.route_log.read_text()
        for item in self.credentials:
            self.assertNotIn(item["api_key"], text)
        self.assertIn("input_sha256", text)

    def test_different_returned_model_is_not_accepted(self):
        response = SimpleNamespace(model="other-model", status="completed", usage=None)
        with patch.object(self.model, "_request", return_value=response):
            with self.assertRaisesRegex(ValueError, "different model"):
                self.model._query(self.messages)

    def test_reasoning_output_metadata_is_omitted_only_on_the_wire(self):
        self.messages[2].update(status="completed", content=None, summary=[])
        before = copy.deepcopy(self.messages)
        with patch.object(self.model, "_request", return_value=self.response) as request:
            self.model._query(self.messages)
        item = request.call_args.kwargs["messages"][1]
        self.assertNotIn("status", item)
        self.assertNotIn("content", item)
        self.assertEqual(item["encrypted_content"], "opaque-state")
        self.assertEqual(item["summary"], [])
        self.assertEqual(self.messages, before)


if __name__ == "__main__":
    unittest.main()
