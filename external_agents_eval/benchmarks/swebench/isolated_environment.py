"""Host-side Docker control; no model credentials or network inside task containers."""
from __future__ import annotations

import asyncio
import io
import shlex
import tarfile
from pathlib import Path
from types import SimpleNamespace


def task_container(client, image: str, *, name: str, cpus: int = 2, memory: str = "6g"):
    return client.containers.create(
        image, command=["sleep", "infinity"], name=name, working_dir="/testbed",
        network_mode="none", cap_drop=["ALL"], security_opt=["no-new-privileges:true"],
        privileged=False, nano_cpus=cpus * 1_000_000_000, mem_limit=memory,
        pids_limit=512, environment={"PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"},
        labels={"external_eval": "v62-verified100"},
    )


class IsolatedEnvironment:
    def __init__(self, container, *, instance_id: str):
        self.container = container
        self.session_id = instance_id + "__sv62"

    def agent_process_env(self, values):
        # These values belong to the host-side model client, never docker exec.
        return dict(values)

    def exec_sync(self, command: str, *, cwd=None, timeout_sec=None, **kwargs):
        timeout = str(max(1, int(timeout_sec or 180)))
        result = self.container.exec_run(
            ["timeout", "--signal=TERM", "--kill-after=5", timeout, "bash", "-lc", command],
            workdir=cwd, demux=True,
        )
        stdout, stderr = result.output or (b"", b"")
        return SimpleNamespace(
            return_code=result.exit_code, stdout=(stdout or b"").decode("utf-8", errors="replace"),
            stderr=(stderr or b"").decode("utf-8", errors="replace"),
        )

    async def exec(self, command, **kwargs):
        return await asyncio.to_thread(self.exec_sync, command, **kwargs)

    async def upload_file(self, source: Path, destination: str):
        def upload():
            data = Path(source).read_bytes()
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                item = tarfile.TarInfo(Path(destination).name)
                item.size = len(data)
                item.mode = 0o644
                archive.addfile(item, io.BytesIO(data))
            self.exec_sync("mkdir -p " + shlex.quote(str(Path(destination).parent)))
            if not self.container.put_archive(str(Path(destination).parent), stream.getvalue()):
                raise RuntimeError("Could not upload a workflow artifact")
        await asyncio.to_thread(upload)

    async def download_file(self, source: str, destination: Path):
        def download():
            chunks, _ = self.container.get_archive(source)
            with tarfile.open(fileobj=io.BytesIO(b"".join(chunks))) as archive:
                item = next(member for member in archive.getmembers() if member.isfile())
                Path(destination).write_bytes(archive.extractfile(item).read())
        await asyncio.to_thread(download)
