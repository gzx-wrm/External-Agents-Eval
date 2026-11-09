"""Mini's native Responses model with host-side, session-sticky endpoint routing."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import openai
from minisweagent.models.litellm_response_model import (
    LitellmResponseModel,
    LitellmResponseModelConfig,
)
from minisweagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API


class RoutedConfig(LitellmResponseModelConfig):
    endpoints_file: Path
    route_log: Path
    initial_endpoint: int = 0


class IncompleteStreamError(RuntimeError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def request_item(message):
    item = dict(message)
    if item.get("type") == "reasoning":
        # SDK output models add an optional status field, which these gateways
        # reject on input. Keep all real reasoning content and the raw trajectory.
        item.pop("status", None)
        item = {key: value for key, value in item.items() if value is not None}
    return item


class RoutedResponseModel(LitellmResponseModel):
    """Keep Mini prompts, tool schema, observation formatting and parsing unchanged."""

    abort_exceptions = LitellmResponseModel.abort_exceptions + [
        openai.BadRequestError, openai.AuthenticationError,
        openai.PermissionDeniedError, openai.NotFoundError,
    ]

    def __init__(self, **kwargs):
        super().__init__(config_class=RoutedConfig, **kwargs)
        self._endpoints = json.loads(self.config.endpoints_file.read_text())
        if len(self._endpoints) != 2 or any(
            not item.get("api_key") or not item.get("base_url") for item in self._endpoints
        ):
            raise ValueError("Exactly two host-side endpoint credentials are required")
        self._current = self.config.initial_endpoint % len(self._endpoints)
        self._lock = threading.Lock()
        self._sequence = 0

    def _record(self, value):
        self.config.route_log.parent.mkdir(parents=True, exist_ok=True)
        with self.config.route_log.open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def _request(self, endpoint, *, instructions, messages, parameters):
        # Credentials are supplied only here, never added to serializable config.
        with openai.OpenAI(base_url=endpoint["base_url"], api_key=endpoint["api_key"],
                           max_retries=0, timeout=600) as client:
            with client.responses.create(
                model=self.config.model_name.removeprefix("openai/"),
                instructions=instructions, input=messages,
                tools=[BASH_TOOL_RESPONSE_API], stream=True, **parameters,
            ) as events:
                final = None
                for event in events:
                    if event.type in {"response.completed", "response.incomplete", "response.failed"}:
                        final = event.response
                if final is None or final.status == "failed":
                    raise IncompleteStreamError("Responses stream has no successful terminal response")
                return final

    def _query(self, messages, **kwargs):
        # An explicit instructions field prevents gateway-specific default prompts.
        systems = [m for m in messages if m.get("role") == "system"]
        if len(systems) != 1 or not isinstance(systems[0].get("content"), str):
            raise ValueError("Expected the single original Mini system message")
        instructions = systems[0]["content"]
        inputs = [request_item(m) for m in messages if m.get("role") != "system"]
        parameters = self.config.model_kwargs | kwargs
        parameters.pop("drop_params", None)
        retryable = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                     openai.InternalServerError, openai.AuthenticationError,
                     openai.PermissionDeniedError, openai.NotFoundError, IncompleteStreamError)
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            # A failed request is retried with identical history, including tool
            # results and opaque reasoning items. Never replay any Bash action.
            for attempt, index in enumerate([self._current, 1 - self._current]):
                endpoint = self._endpoints[index]
                audit = {"query": sequence, "attempt": attempt + 1,
                         "endpoint": endpoint["id"], "input_items": len(inputs),
                         "input_sha256": fingerprint(inputs),
                         "instructions_sha256": fingerprint(instructions)}
                try:
                    response = self._request(endpoint, instructions=instructions,
                                             messages=inputs, parameters=parameters)
                except Exception as error:
                    self._record({**audit, "status": "error", "error_type": type(error).__name__,
                                  "http_status": getattr(error, "status_code", None)})
                    if not isinstance(error, retryable) or attempt == 1:
                        raise
                    continue
                expected = self.config.model_name.removeprefix("openai/")
                if not response.model.startswith(expected):
                    raise ValueError("The endpoint returned a different model")
                self._current = index
                usage = response.usage.model_dump(mode="json") if response.usage else None
                self._record({**audit, "status": "success", "model": response.model,
                              "response_status": response.status, "usage": usage})
                return response
