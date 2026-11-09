"""V6.4 DeepSWE runner pieces that do not need Docker or model calls."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts/deepswe_v64"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location("v64_" + name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ledger_input = load("ledger_input_v64")
gate_executor = load("gate_executor")

from external_agents_eval.workflows.solve_review_repair.build_gate import (  # noqa: E402
    ERROR,
    FAIL,
    PASS,
    TIMEOUT,
    GateCommand,
)

LEDGER = {"summary": "s", "obligations": [{"id": "O001", "content": {"statement": "Do it."}}],
          "preservation_obligations": [], "open_questions": []}


def test_solver_supplement_is_english_single_and_complete():
    task = {"task_id": "t", "instruction": "Original issue.\n", "image": "img"}
    out = ledger_input.augment_task(task, LEDGER)
    assert out["instruction"].startswith(task["instruction"]) and task["instruction"] == "Original issue.\n"
    assert out["instruction"].count(ledger_input.HEADING) == 1
    assert out["instruction"].count(ledger_input.GUIDANCE) == 1
    assert not re.search(r"[一-鿿]", out["instruction"])
    block = out["instruction"].split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    assert json.loads(block) == LEDGER
    for word in ("obligations", "preservation_obligations", "open_questions", "required",
                 "advisory", "authoritative"):
        assert word in ledger_input.GUIDANCE


def test_frozen_ledger_hashes_are_enforced(tmp_path):
    data = json.dumps(LEDGER).encode()
    path = tmp_path / "ledger.json"
    path.write_bytes(data)
    kwargs = {"file_sha256": hashlib.sha256(data).hexdigest(),
              "ledger_sha256": ledger_input.canonical_sha256(LEDGER)}
    assert ledger_input.load_frozen_ledger(path, **kwargs) == LEDGER
    path.write_bytes(data + b" ")
    with pytest.raises(ValueError, match="file changed"):
        ledger_input.load_frozen_ledger(path, **kwargs)


class FakeContainer:
    def __init__(self, script):
        self.script, self.calls, self.uploads, self.removed = script, [], {}, False

    def start(self):
        pass

    def exec_run(self, command, workdir=None, demux=False):
        self.calls.append(command)
        inner = command[-1]
        for pattern, (code, output) in self.script.items():
            if pattern in inner:
                return SimpleNamespace(exit_code=code, output=output.encode())
        return SimpleNamespace(exit_code=0, output=b"")

    def put_archive(self, parent, data):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                self.uploads[parent + "/" + member.name] = archive.extractfile(member).read()
        return True

    def remove(self, force=False):
        self.removed = True


class FakeClient:
    def __init__(self, script):
        self.script, self.created, self.closed = script, [], 0
        self.containers = SimpleNamespace(create=self.create)

    def create(self, image, **kwargs):
        container = FakeContainer(self.script)
        self.created.append((image, kwargs, container))
        return container

    def close(self):
        self.closed += 1


def executor(client, tmp_path=None):
    return gate_executor.DockerGateExecutor(lambda: client, image_id="sha256:img", cpus=2,
                                            memory_mb=8192, task_id="t", log_dir=tmp_path)


def test_gate_containers_are_offline_bounded_and_removed(tmp_path):
    client = FakeClient({"git ls-files -z": (0, "Cargo.toml\0core/Cargo.toml\0core/src/lib.rs\0"),
                         "cat -- core/Cargo.toml": (0, '[package]\nname = "core"\n'),
                         "cargo test": (101, "error[E0282]: type annotations needed")})
    gate = executor(client, tmp_path / "logs")
    import asyncio
    files = asyncio.run(gate.list_files())
    assert files == ["Cargo.toml", "core/Cargo.toml", "core/src/lib.rs"]
    assert asyncio.run(gate.read_file("core/Cargo.toml")).startswith("[package]")
    command = GateCommand("rust", ".", "cargo test --no-run --offline --quiet -p core")
    outcomes = asyncio.run(gate.run("P0", b"diff --git a/x b/x\n", [command], 600))
    assert outcomes[0].status == FAIL and "E0282" in outcomes[0].log_tail
    image, kwargs, container = client.created[-1]
    assert image == "sha256:img" and kwargs["network_mode"] == "none"
    assert kwargs["nano_cpus"] == 2_000_000_000 and kwargs["mem_limit"] == "8192m"
    assert container.removed and all(c.removed for _, _, c in client.created)
    assert container.uploads["/tmp/v64-gate.patch"] == b"diff --git a/x b/x\n"
    wrapped = container.calls[-1]
    assert wrapped[:3] == ["timeout", "--signal=TERM", "--kill-after=10"] and wrapped[4:6] == ["bash", "-lc"]
    assert "CARGO_NET_OFFLINE=true" in wrapped[-1] and "GOPROXY=off" in wrapped[-1]
    assert (tmp_path / "logs/P0-1.log").read_text().startswith("$ cd .")


def test_gate_patch_apply_failure_and_timeouts_are_not_failures(tmp_path):
    client = FakeClient({"git apply": (1, "error: patch failed"), "go test": (124, "")})
    gate = executor(client)
    import asyncio
    command = GateCommand("go", ".", "go test -count=1 -run '^$' ./...")
    outcome = asyncio.run(gate.run("P0", b"patch", [command], 600))[0]
    assert outcome.status == ERROR and "patch did not apply" in outcome.log_tail
    client = FakeClient({"go test": (124, "")})
    outcome = asyncio.run(executor(client).run("base", b"", [command], 600))[0]
    assert outcome.status == TIMEOUT
    client = FakeClient({"go test": (0, "ok")})
    assert asyncio.run(executor(client).run("base", b"", [command], 600))[0].status == PASS


def test_runner_freezes_existing_sources_and_scores_p0_only_when_different():
    pytest.importorskip("minisweagent")
    runner = load("run")
    hashes = runner.code_hashes()
    assert all((PROJECT / name).is_file() for name in runner.FROZEN_SOURCES)
    assert runner.SKILL_DIR + "/" in hashes and len(hashes) == len(runner.FROZEN_SOURCES) + 1
    assert runner.analysis_plan(b"same", b"same") == "reuse_final_score"
    assert runner.analysis_plan(b"p0", b"p1") == "score_p0_separately"
    assert runner.EXPERIMENT_ID == "v6.4"
