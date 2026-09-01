from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """Stable identity and execution contract for a benchmark backend."""

    benchmark_id: str
    revision: str
    repository_url: str
    workdir: str
