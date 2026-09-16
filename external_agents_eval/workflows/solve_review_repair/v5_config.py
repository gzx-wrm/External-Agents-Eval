from __future__ import annotations

from dataclasses import dataclass

V5_PROMPT_VERSION = "v5-5"
V5_CONTROLLER_VERSION = "v5-4"


@dataclass(frozen=True)
class V5Config:
    batch_size: int = 3
    max_rounds: int = 64
    max_format_retries: int = 2
    max_semantic_retries: int = 1
    max_no_progress: int = 2
    lane_deadline_sec: float = 45 * 60
    role_timeout_sec: float = 30 * 60
    no_compaction: bool = True
    model: str = "deepseek-v4-flash"
    prompt_version: str = V5_PROMPT_VERSION
    controller_version: str = V5_CONTROLLER_VERSION

    def validate(self) -> None:
        if self.batch_size < 1 or self.max_rounds < 1:
            raise ValueError("V5 batch_size and max_rounds must be positive")
        if self.max_format_retries < 0 or self.max_semantic_retries < 0:
            raise ValueError("V5 retry limits cannot be negative")
        if self.max_no_progress < 1 or self.lane_deadline_sec <= 0 or self.role_timeout_sec <= 0:
            raise ValueError("V5 deadlines and no-progress limit must be positive")
