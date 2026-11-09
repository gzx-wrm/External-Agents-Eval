"""Run V6.4 build-gate commands in fresh, offline containers of the task image.

The container is created from the unchanged task image (its /app is the image's
environment baseline), never sees the model, hidden tests or grader files, and is
removed after each run. Candidate patches are deltas from that baseline.
"""
from __future__ import annotations

import asyncio
import io
import posixpath
import shlex
import tarfile
import time
import uuid
from pathlib import Path

from external_agents_eval.workflows.solve_review_repair.build_gate import (
    ERROR,
    TIMEOUT,
    CommandOutcome,
    GateCommand,
    classify,
    error_codes,
    tail,
)

INSPECTED_FILES = {"Cargo.toml", "pyproject.toml", "setup.cfg", "tox.ini"}
KILL_AFTER_SEC = 10


class DockerGateExecutor:
    def __init__(self, client_factory, *, image_id, cpus, memory_mb, task_id,
                 workdir="/app", log_dir: Path | None = None):
        self.client_factory = client_factory
        self.image_id = image_id
        self.cpus = int(cpus)
        self.memory_mb = int(memory_mb)
        self.task_id = task_id
        self.workdir = workdir
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._files: list[str] | None = None
        self._contents: dict[str, str | None] = {}

    # -- container helpers -------------------------------------------------
    def _create(self, client):
        container = client.containers.create(
            self.image_id, name="v64-gate-" + uuid.uuid4().hex[:12], entrypoint=[],
            command=["sleep", "infinity"], working_dir=self.workdir, network_mode="none",
            nano_cpus=self.cpus * 1_000_000_000, mem_limit=f"{self.memory_mb}m",
            labels={"external_eval": "v64-build-gate", "task": self.task_id},
        )
        container.start()
        return container

    def _exec(self, container, command: str, timeout_sec: float):
        seconds = str(max(1, int(timeout_sec)))
        result = container.exec_run(
            ["timeout", "--signal=TERM", f"--kill-after={KILL_AFTER_SEC}", seconds,
             "bash", "-lc", command],
            workdir=self.workdir, demux=False,
        )
        output = result.output or b""
        return result.exit_code, output.decode("utf-8", errors="replace")

    @staticmethod
    def _upload(container, data: bytes, destination: str):
        parent, name = posixpath.split(destination)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            archive.addfile(item, io.BytesIO(data))
        if not container.put_archive(parent, stream.getvalue()):
            raise RuntimeError("Build gate patch upload failed")

    # -- GateExecutor protocol ---------------------------------------------
    def _inspect_sync(self):
        client = self.client_factory()
        container = None
        try:
            container = self._create(client)
            code, out = self._exec(container, "git ls-files -z", 120)
            if code != 0:
                raise RuntimeError("Could not list repository files for the build gate")
            files = [path for path in out.split("\0") if path]
            contents = {}
            for path in files:
                if posixpath.basename(path) in INSPECTED_FILES:
                    code, text = self._exec(container, "cat -- " + shlex.quote(path), 60)
                    contents[path] = text if code == 0 else None
            return files, contents
        finally:
            if container is not None:
                container.remove(force=True)
            client.close()

    async def list_files(self) -> list[str]:
        if self._files is None:
            self._files, self._contents = await asyncio.to_thread(self._inspect_sync)
        return list(self._files)

    async def read_file(self, path: str) -> str | None:
        if self._files is None:
            await self.list_files()
        return self._contents.get(path)

    def _run_sync(self, label, patch, commands, timeout_sec):
        client = self.client_factory()
        container = None
        outcomes: list[CommandOutcome] = []
        try:
            container = self._create(client)
            if patch:
                self._upload(container, patch, "/tmp/v64-gate.patch")
                code, out = self._exec(
                    container, "git apply --binary --whitespace=nowarn /tmp/v64-gate.patch", 120)
                if code != 0:
                    message = "patch did not apply in the gate container:\n" + out
                    return [CommandOutcome(ERROR, code, 0.0, tail(message)) for _ in commands]
            deadline = time.monotonic() + timeout_sec
            for index, command in enumerate(commands):
                remaining = deadline - time.monotonic()
                if remaining < 1:
                    outcomes.append(CommandOutcome(TIMEOUT, None, 0.0, "run budget exhausted"))
                    continue
                started = time.monotonic()
                code, out = self._exec(container, command.shell(), remaining)
                duration = round(time.monotonic() - started, 3)
                timed_out = duration >= remaining - 0.5 and code != 0
                outcomes.append(CommandOutcome(classify(code, timed_out=timed_out), code,
                                               duration, tail(out), error_codes(out)))
                if self.log_dir is not None:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    (self.log_dir / f"{label}-{index + 1}.log").write_text(
                        f"$ {command.shell()}\n# exit={code} duration={duration}\n{out}",
                        encoding="utf-8")
            return outcomes
        finally:
            if container is not None:
                container.remove(force=True)
            client.close()

    async def run(self, label: str, patch: bytes, commands: list[GateCommand],
                  timeout_sec: float) -> list[CommandOutcome]:
        return await asyncio.to_thread(self._run_sync, label, patch, commands, timeout_sec)
