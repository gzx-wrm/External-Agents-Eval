"""V6.6 transport retry: resend a model request only when no valid response arrived.

Retried: HTTP 429/5xx, connection or timeout errors, and a 200 whose body is empty,
``null``, not JSON or has no ``choices`` (observed during provider overload).
Never retried: any response that carries ``choices``, whatever its content, and
other HTTP errors. Sampling is therefore never repeated on a delivered output.
"""
from __future__ import annotations

import json
import time
import types
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from . import ft_inference
from .artifacts import ArtifactStore
from .ft_inference import FTLedgerClient

RETRY_WAITS_SEC = (30, 60, 120)
RETRY_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


class NoValidResponse(Exception):
    """A 200 reply that carries no model output."""


class BufferedResponse:
    """The part of urlopen's response object the clients use."""

    def __init__(self, status: int, data: bytes) -> None:
        self.status, self._data = status, data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "BufferedResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def open_with_retry(request: Any, *, timeout: float, record: Callable[[dict[str, Any]], None],
                    waits: tuple[float, ...] = RETRY_WAITS_SEC, sleep: Callable[[float], None] = time.sleep,
                    opener: Callable[..., Any] = urlopen) -> BufferedResponse:
    last: BaseException | None = None
    for index in range(len(waits) + 1):
        entry: dict[str, Any] = {"attempt": index + 1,
                                 "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            with opener(request, timeout=timeout) as response:
                data, status = response.read(), response.status
            text = data.decode("utf-8", errors="replace").strip()
            if not text or text == "null":
                raise NoValidResponse("empty or null response body")
            try:
                payload = json.loads(text)
            except ValueError as error:
                raise NoValidResponse("response body is not JSON") from error
            if not isinstance(payload, dict) or not payload.get("choices"):
                raise NoValidResponse("response body has no choices")
            entry.update(outcome="response", http_status=status)
            record(entry)
            return BufferedResponse(status, data)
        except HTTPError as error:
            entry.update(outcome="http_error", http_status=error.code)
            if error.code not in RETRY_HTTP_STATUS:
                record(entry)
                raise
            last = error
        except (NoValidResponse, URLError, OSError) as error:
            entry.update(outcome="no_valid_response", error=f"{type(error).__name__}: {error}"[:300])
            last = error
        record(entry)
        if index < len(waits):
            sleep(waits[index])
    assert last is not None
    raise last


class V66FTLedgerClient(FTLedgerClient):
    """FTLedgerClient with the identical request/validation code and the retrying transport.

    ``_generate`` runs FTLedgerClient's own code object; only its ``urlopen`` global is
    replaced, so the FT protocol cannot drift from the frozen client.
    """

    def __init__(self, *, model: str, base_url: str, api_key: str,
                 waits: tuple[float, ...] = RETRY_WAITS_SEC,
                 sleep: Callable[[float], None] = time.sleep, opener: Callable[..., Any] = urlopen) -> None:
        super().__init__(model=model, base_url=base_url, api_key=api_key)
        self._waits, self._sleep, self._opener = waits, sleep, opener
        self._artifacts: ArtifactStore | None = None
        self._transport: list[dict[str, Any]] = []
        frozen = FTLedgerClient._generate
        scope = dict(vars(ft_inference))
        scope["urlopen"] = self._open
        self._frozen_generate = types.FunctionType(frozen.__code__, scope, frozen.__name__,
                                                   frozen.__defaults__, frozen.__closure__)

    def _open(self, request: Any, timeout: float) -> BufferedResponse:
        def record(entry: dict[str, Any]) -> None:
            self._transport.append(entry)
            if self._artifacts is not None:
                self._artifacts.write_json("v6.2/ledger/transport.json", self._transport)

        return open_with_retry(request, timeout=timeout, record=record, waits=self._waits,
                               sleep=self._sleep, opener=self._opener)

    def _generate(self, sample: dict[str, Any], artifacts: ArtifactStore):
        self._artifacts = artifacts
        return self._frozen_generate(self, sample, artifacts)
