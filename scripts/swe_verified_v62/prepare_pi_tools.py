"""Provision pinned Pi file tools on the host; task containers remain offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from external_agents_eval.agents.pi_file_tools import PI_VERSION, runtime_directory

NODE_VERSION = "22.19.0"


def run(*args, env=None):
    return subprocess.check_output(args, env=env, text=True).strip()


def prepare(runtime: Path):
    runtime.mkdir(parents=True, exist_ok=True)
    binary = runtime / "bin"
    binary.mkdir(exist_ok=True)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ.get("PATH", ""),
                       PI_CODING_AGENT_DIR=str(runtime / "agent"))
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node and npm:
        shutil.copy2(Path(node).resolve(), binary / "node")
    else:
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
            raise RuntimeError("Install Node/npm first on this host")
        arch = "x64" if platform.machine() == "x86_64" else "arm64"
        name = f"node-v{NODE_VERSION}-linux-{arch}"
        url = f"https://nodejs.org/dist/v{NODE_VERSION}/{name}.tar.xz"
        with tempfile.TemporaryDirectory() as temporary:
            download = Path(temporary) / "node.tar.xz"
            urllib.request.urlretrieve(url, download)
            checksums = urllib.request.urlopen(
                f"https://nodejs.org/dist/v{NODE_VERSION}/SHASUMS256.txt", timeout=60,
            ).read().decode()
            expected = next(line.split()[0] for line in checksums.splitlines()
                            if line.endswith(name + ".tar.xz"))
            if hashlib.sha256(download.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Node archive checksum mismatch")
            with tarfile.open(download) as archive:
                archive.extractall(temporary, filter="data")
            shutil.copy2(Path(temporary) / name / "bin/node", binary / "node")
            shutil.copytree(Path(temporary) / name / "lib", runtime / "lib", dirs_exist_ok=True)
        npm = str(runtime / "lib/node_modules/npm/bin/npm-cli.js")
    subprocess.run([str(binary / "node"), npm, "install", "--prefix", str(runtime),
                    "--ignore-scripts", "--omit=dev", "--no-audit", "--no-fund",
                    f"@earendil-works/pi-coding-agent@{PI_VERSION}"], env=environment, check=True)
    helper = Path(__file__).resolve().parents[2] / "external_agents_eval/agents/pi_file_tools.mjs"
    tools = json.loads(run(str(binary / "node"), str(helper), str(runtime), "--prepare", env=environment))
    for name, tool in tools.items():
        source = Path(shutil.which(tool, path=environment["PATH"]) or tool)
        destination = binary / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    schemas = run(str(binary / "node"), str(helper), str(runtime), "--schemas", env=environment)
    (runtime / "tool-schemas.json").write_text(schemas + "\n")
    with tarfile.open(runtime / "runtime.tar.gz", "w:gz") as archive:
        for name in ("bin", "node_modules"):
            archive.add(runtime / name, arcname=name)
    report = {
        "pi_version": PI_VERSION, "node_version": run(str(binary / "node"), "--version"),
        "rg_version": run(str(binary / "rg"), "--version").splitlines()[0],
        "fd_version": run(str(binary / "fd"), "--version"),
        "schemas_sha256": hashlib.sha256((runtime / "tool-schemas.json").read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((runtime / "package-lock.json").read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256((runtime / "runtime.tar.gz").read_bytes()).hexdigest(),
        "platform": platform.system(), "machine": platform.machine(),
        "runtime": str(runtime), "model_api_calls": 0,
    }
    (runtime / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=runtime_directory())
    prepare(parser.parse_args().runtime.resolve())
