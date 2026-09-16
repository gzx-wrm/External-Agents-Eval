from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import sha256_bytes


class PromptRenderer:
    PLACEHOLDER = re.compile(r"{{([^{}]+)}}")

    def __init__(self, prompt_dir: str | Path) -> None:
        self.prompt_dir = Path(prompt_dir)

    @staticmethod
    def json_text(payload: Any) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def load_few_shots(paths: tuple[Path, ...]) -> str:
        if not paths:
            return "No few-shot examples are configured for this development run."
        examples: list[str] = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            role = "OUTPUT" if path.name.endswith("_output.json") else "INPUT OR CONTEXT"
            examples.append(
                f"EXAMPLE {role} ARTIFACT: {path.parent.name}/{path.name}\n"
                "```json\n"
                + json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n```"
            )
        return "\n\n".join(examples)

    def render(self, template_name: str, replacements: dict[str, str]) -> tuple[str, str]:
        path = self.prompt_dir / template_name
        template = path.read_text(encoding="utf-8")
        placeholders = tuple(dict.fromkeys(self.PLACEHOLDER.findall(template)))
        unresolved = [name for name in placeholders if name not in replacements]
        if unresolved:
            raise ValueError("Unresolved prompt placeholders: " + ", ".join(unresolved))
        rendered = self.PLACEHOLDER.sub(
            lambda match: replacements[match.group(1)],
            template,
        )
        return rendered, sha256_bytes(rendered.encode("utf-8"))
