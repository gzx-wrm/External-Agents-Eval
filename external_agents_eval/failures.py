from __future__ import annotations

import re
from enum import Enum


class FailureKind(str, Enum):
    EXTERNAL_NETWORK = "external_network"
    LLM_SERVICE = "llm_service"
    AGENT = "agent"
    UNKNOWN = "unknown"


_NETWORK_PATTERNS = (
    r"\b(?:502|503|504)\b",
    r"read\s*timeout",
    r"connect(?:ion)?\s*(?:timeout|timed out|reset|refused)",
    r"temporary failure in name resolution",
    r"name or service not known",
    r"tls|ssl.*(?:eof|error)",
    r"max retries exceeded",
    r"docker.*(?:pull|download).*(?:failed|error|timeout)",
    r"pypi|hugging\s*face",
)
_LLM_PATTERNS = (
    r"\b429\b",
    r"rate.?limit",
    r"api\.deepseek\.com.*\b5\d\d\b",
    r"deepseek.*(?:timeout|timed out|connection|server|5\d\d)",
    r"(?:openai|llm).*(?:timeout|timed out|connection|server|5\d\d)",
)


def classify_failure(text: str) -> FailureKind:
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in _LLM_PATTERNS):
        return FailureKind.LLM_SERVICE
    if any(re.search(pattern, lowered) for pattern in _NETWORK_PATTERNS):
        return FailureKind.EXTERNAL_NETWORK
    if any(
        marker in lowered
        for marker in (
            "agent process exited",
            "did not produce",
            "invalid tool",
            "framework error",
            "empty response",
        )
    ):
        return FailureKind.AGENT
    return FailureKind.UNKNOWN
