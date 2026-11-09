"""Single-attempt, tool-free FT inference using the original request protocol."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

from .artifacts import ArtifactStore, sha256_json
from .ft_protocol import validate_output
from .models import IndeterminateInvocationError

FT_RESOURCES = Path(__file__).with_name("ft")
FT_SYSTEM_PROMPT = (FT_RESOURCES / "prompt.txt").read_text(encoding="utf-8").strip()
FT_SCHEMA = json.loads((FT_RESOURCES / "spec_output_schema.json").read_text(encoding="utf-8"))


class FTLedgerClient:
    def __init__(self, *, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def generate(
        self, sample: dict[str, Any], *, artifacts: ArtifactStore,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.to_thread(self._generate, sample, artifacts)

    def _generate(
        self, sample: dict[str, Any], artifacts: ArtifactStore,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prefix = "v6.2/ledger"
        request = {
            "model": self.model, "messages": sample["messages"],
            "thinking": {"type": "enabled"}, "reasoning_effort": "high",
            "max_tokens": 65536, "seed": 42, "stream": False,
        }
        request_path = artifacts.root / prefix / "request.json"
        result_path = artifacts.root / prefix / "result.json"
        if request_path.exists():
            if json.loads(request_path.read_text()) != request:
                raise ValueError("FT request changed; use a new run for different inputs")
            if not result_path.exists():
                raise IndeterminateInvocationError(
                    "FT request exists without a completed result; do not automatically resample"
                )
            result = json.loads(result_path.read_text())
            if result["status"] not in {"passed", "invalid_output"}:
                raise ValueError(f"Prior FT attempt ended with {result['status']}; use a new run")
            content = (artifacts.root / prefix / "output.txt").read_text(encoding="utf-8")
            return self._validated(content, sample)

        artifacts.write_json(f"{prefix}/request.json", request)
        started_at = datetime.now(timezone.utc).isoformat()
        result = {
            "task_id": sample["task_id"], "base_commit": sample["base_commit"],
            "status": "running", "started_at": started_at, "attempts": 1, "usage": {},
            "messages_sha256": sha256_json(sample["messages"]),
        }
        started = time.monotonic()
        try:
            http = Request(
                self.base_url + "/chat/completions",
                data=json.dumps(request, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.api_key},
            )
            # FT used one request, 1800 seconds, without transport or format retries.
            with urlopen(http, timeout=1800) as response:
                raw = response.read().decode()
                result["http_status"] = response.status
            artifacts.write_text(f"{prefix}/raw_response.txt", raw)
            response = json.loads(raw)
            artifacts.write_json(f"{prefix}/response.json", response)
            choice = response["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning_content") or ""
            content = message.get("content")
            result.update({
                "usage": response.get("usage") or {},
                "returned_model": response.get("model"), "response_id": response.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "thinking_observed": bool(reasoning), "reasoning_chars": len(reasoning),
            })
            artifacts.write_text(f"{prefix}/reasoning.txt", reasoning)
            if not isinstance(content, str):
                raise ValueError("Response contains no final text content")  # noqa: TRY004 - FT protocol
            artifacts.write_text(f"{prefix}/output.txt", content)
            payload, checks = validate_output(content, sample, Draft202012Validator(FT_SCHEMA))
            artifacts.write_json(f"{prefix}/validation.json", checks)
            result["validation"] = checks
            if payload is not None:
                artifacts.write_json(f"{prefix}/spec_output.json", payload)
            if choice.get("finish_reason") == "length":
                result["status"] = "truncated"
            elif choice.get("finish_reason") != "stop":
                result["status"] = "incomplete"
            else:
                result["status"] = "passed" if checks["sources_valid"] else "invalid_output"
            usage = result["usage"]
            cached = usage.get("prompt_cache_hit_tokens")
            if cached is None:
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            metrics = {
                "total_prompt_tokens": usage.get("prompt_tokens", 0),
                "total_completion_tokens": usage.get("completion_tokens", 0),
                "total_cached_tokens": cached,
            }
            artifacts.write_json(f"{prefix}/trajectory.json", {
                "schema_version": "ATIF-v1.7", "session_id": sample["messages_sha256"],
                "agent": {"name": "ft-ledger", "version": "v6.2", "model_name": self.model},
                "steps": [
                    {"step_id": 1, "source": "user", "message": sample["messages"][1]["content"]},
                    {"step_id": 2, "source": "agent", "message": content,
                     "reasoning_content": reasoning, "model_name": result["returned_model"] or self.model},
                ],
                "final_metrics": {"total_steps": 2, **metrics},
            })
        except HTTPError as error:
            result.update({"status": "request_error", "http_status": error.code,
                           "error_type": "HTTPError"})
            artifacts.write_text(f"{prefix}/http-error.txt", error.read().decode(errors="replace"))
        except Exception as error:  # noqa: BLE001 - persist the single FT attempt's outcome
            result.update({"status": "request_error", "error_type": type(error).__name__,
                           "error": artifacts.redact(str(error))})
        result.update({"finished_at": datetime.now(timezone.utc).isoformat(),
                       "elapsed_seconds": round(time.monotonic() - started, 3)})
        artifacts.write_json(f"{prefix}/result.json", result)
        if result["status"] not in {"passed", "invalid_output"}:
            raise ValueError(f"FT ledger generation ended with {result['status']}")
        return self._validated(content, sample)

    @staticmethod
    def _validated(content: str, sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, checks = validate_output(content, sample, Draft202012Validator(FT_SCHEMA))
        if not checks["schema_valid"]:
            raise ValueError(f"FT ledger is structurally invalid: {checks['errors']}")
        return payload, checks
