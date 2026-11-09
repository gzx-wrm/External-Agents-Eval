"""Transport Pi 0.84.2 file tools into mini; evidence remains structured read observations."""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

PI_VERSION = "0.84.2"
PI_TOOL_NAMES = ("read", "grep", "find", "ls")
PI_EVIDENCE_PROTOCOL = "pi-read-observed-v1"
REMOTE_RUNTIME = "/external_eval/pi-file-tools"
REMOTE_HELPER = "/external_eval/pi_file_tools.mjs"


def runtime_directory() -> Path:
    return Path(os.environ.get(
        "EXTERNAL_EVAL_PI_TOOLS", str(Path.home() / ".cache/external_agents_eval/pi-file-tools"),
    )) / PI_VERSION


def load_tool_schemas(runtime: Path) -> list[dict]:
    return json.loads((runtime / "tool-schemas.json").read_text())


async def prepare_tools(environment) -> Path:
    runtime = runtime_directory()
    archive = runtime / "runtime.tar.gz"
    if not archive.is_file():
        raise RuntimeError("Pi file-tool runtime is missing; run scripts/swe_verified_v62/prepare_pi_tools.py")
    metadata = json.loads((runtime / "runtime.json").read_text())
    if metadata.get("platform") != "Linux" or metadata.get("machine") != "x86_64":
        raise RuntimeError("Verified x86_64 containers require the runtime prepared on a Linux x86_64 host")
    await environment.upload_file(archive, REMOTE_RUNTIME + ".tar.gz")
    await environment.upload_file(Path(__file__).with_suffix(".mjs"), REMOTE_HELPER)
    result = await environment.exec(
        f"mkdir -p {REMOTE_RUNTIME} && tar -xzf {REMOTE_RUNTIME}.tar.gz -C {REMOTE_RUNTIME}"
        f" && rm -f {REMOTE_RUNTIME}.tar.gz",
    )
    if result.return_code:
        raise RuntimeError("Could not provision Pi file tools: " + result.stderr)
    return runtime


def execute_tool(environment, cwd: str, name: str, arguments: dict, tool_call_id: str) -> dict:
    request = json.dumps({"name": name, "arguments": arguments, "cwd": cwd,
                          "tool_call_id": tool_call_id}, ensure_ascii=False)
    command = " ".join(shlex.quote(item) for item in (
        REMOTE_RUNTIME + "/bin/node", REMOTE_HELPER, REMOTE_RUNTIME, request,
    ))
    raw = environment.exec_sync(command, cwd=cwd)
    if raw.return_code:
        raise RuntimeError("Pi file-tool process failed: " + raw.stderr)
    return json.loads(raw.stdout)
