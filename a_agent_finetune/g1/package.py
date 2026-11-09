"""Bundle only frozen G1 inputs and source; exclude weights, environments and run outputs."""

from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

from .materials import file_sha256, verify_materials

HERE = Path(__file__).resolve().parent


def main() -> None:
    verify_materials(HERE)
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    files = sorted(set(manifest["files_sha256"]) | {"data/manifest.json", "checks/report.json"})
    destination = HERE / f"dist/g1_materials_v{manifest['material_version']}.tar.gz"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError("Preserve the existing G1 bundle; use a new material version")
    with destination.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as archive:
            for relative in files:
                path = HERE / relative
                info = archive.gettarinfo(str(path), arcname="a_agent_finetune/g1/" + relative)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    digest = file_sha256(destination)
    destination.with_suffix(".sha256").write_text(digest + "  " + destination.name + "\n")
    print(json.dumps({"path": str(destination), "bytes": destination.stat().st_size, "sha256": digest}))


if __name__ == "__main__":
    main()
