"""Held-out29 v6.3 runner pieces that need neither Docker nor model calls."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts/deepswe_v63_heldout/run.py"
PROXY = PROJECT / "scripts/deepswe_v63_heldout/egress_proxy.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load("v63h_runner", RUNNER)


@pytest.fixture
def proxy(monkeypatch):
    monkeypatch.setenv("ALLOWLIST_DOMAINS", "api.deepseek.com")
    monkeypatch.setenv("PROXY_TOKEN", "tok-123")
    monkeypatch.setenv("PROXY_PORT", "8080")
    return load("v63h_egress_proxy", PROXY)


def ask(proxy, request: bytes):
    client, server = socket.socketpair()
    worker = threading.Thread(target=proxy.handle, args=(server, ("127.0.0.1", 1)))
    worker.start()
    client.sendall(request)
    data = b""
    client.settimeout(5)
    try:
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    client.close()
    worker.join(5)
    return data


def auth(token="tok-123"):
    return b"Proxy-Authorization: Basic " + base64.b64encode(b"agent:" + token.encode()) + b"\r\n"


def test_frozen_sources_exist_and_are_hashed():
    for name in run.FROZEN_SOURCES:
        assert (PROJECT / name).is_file(), name
    assert set(run.code_hashes()) == set(run.FROZEN_SOURCES)
    assert "scripts/deepswe_ft_m1_review/run.py" in run.FROZEN_SOURCES
    assert "scripts/deepswe_ft_m1/ledger_input.py" in run.FROZEN_SOURCES


@pytest.mark.parametrize("request_bytes, status", [
    (b"CONNECT api.deepseek.com:443 HTTP/1.1\r\n\r\n", b"407"),
    (b"CONNECT api.deepseek.com:443 HTTP/1.1\r\n" + auth("wrong") + b"\r\n", b"407"),
    (b"CONNECT github.com:443 HTTP/1.1\r\n" + auth() + b"\r\n", b"403"),
    (b"CONNECT api.deepseek.com:80 HTTP/1.1\r\n" + auth() + b"\r\n", b"403"),
    (b"GET http://api.deepseek.com/ HTTP/1.1\r\n" + auth() + b"\r\n", b"403"),
])
def test_proxy_denies_everything_outside_the_policy(proxy, request_bytes, status):
    assert ask(proxy, request_bytes).startswith(b"HTTP/1.1 " + status)


def test_proxy_tunnels_only_the_allowlisted_api(proxy, monkeypatch):
    upstream_client, upstream_server = socket.socketpair()
    targets = []

    def connect(address, timeout=None):
        targets.append(address)
        return upstream_client

    monkeypatch.setattr(proxy.socket, "create_connection", connect)

    def echo():
        data = upstream_server.recv(4096)
        upstream_server.sendall(b"echo:" + data)
        upstream_server.close()

    threading.Thread(target=echo).start()
    response = ask(proxy, b"CONNECT api.deepseek.com:443 HTTP/1.1\r\n" + auth() + b"\r\nhello")
    assert targets == [("api.deepseek.com", 443)]
    assert response == b"HTTP/1.1 200 Connection established\r\n\r\necho:hello"


def test_proxy_environment_matches_pier_filtered_egress():
    sandbox = object.__new__(run.ASandbox)
    sandbox.token = "abc_DEF-123"
    url = "http://agent:abc_DEF-123@pier-egress-proxy:8080"
    assert sandbox.proxy_env == {
        "HTTP_PROXY": url, "HTTPS_PROXY": url, "http_proxy": url, "https_proxy": url,
        "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1",
    }
    try:
        from pier.environments.agent_setup import proxy_environment
    except ImportError:
        return
    assert sandbox.proxy_env == proxy_environment("abc_DEF-123", "pier-egress-proxy", 8080)
    assert sandbox.redact("x abc_DEF-123 y") == "x [REDACTED] y"


def test_pi_environment_uses_pier_exec_semantics():
    calls = []

    class Container:
        def exec_run(self, command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(exit_code=0, output=(b"out", b"err"))

    environment = run.PiContainerEnvironment(Container(), instance_id="t", proxy_env={"HTTPS_PROXY": "p"})
    merged = environment.agent_process_env({"DEEPSEEK_API_KEY": "k"})
    assert merged == {"HTTPS_PROXY": "p", "DEEPSEEK_API_KEY": "k"}
    result = environment.exec_sync("echo hi", cwd="/app", env={"A": "1"}, timeout_sec=12, user="root")
    command, kwargs = calls[0]
    assert command == ["timeout", "--signal=TERM", "--kill-after=5", "12", "bash", "-c", "echo hi"]
    assert kwargs == {"workdir": "/app", "environment": {"A": "1"}, "user": "root", "demux": True}
    assert (result.return_code, result.stdout, result.stderr) == (0, "out", "err")
    environment.exec_sync("true")
    assert calls[1][0][3] == str(run.DEFAULT_EXEC_TIMEOUT_SEC)
    assert calls[1][1]["user"] == ""


def test_pi_env_is_the_historical_v6_agent_env():
    assert run.PI_ENV["PI_OFFLINE"] == "1"
    assert run.PI_ENV["PATH"].startswith("/opt/pi-runtime/bin:")
    assert run.A_ROLE_TIMEOUT_SEC == 1800


def test_egress_probe_renders_valid_python():
    rendered = run.EGRESS_PROBE % {"alias": run.EGRESS_ALIAS, "port": run.EGRESS_PORT,
                                   "host": "api.deepseek.com"}
    body = rendered.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(body, "probe", "exec")
    assert "f\"" not in body and "f'" not in body  # task images ship Python 3.5


def test_egress_check_requires_every_denial():
    good = {"direct_blocked": True, "allowed": "HTTP/1.1 200 Connection established",
            "other_host": "HTTP/1.1 403 Forbidden", "other_port": "HTTP/1.1 403 Forbidden",
            "no_auth": "HTTP/1.1 407 Proxy Authentication Required",
            "hidden_tests_present": False, "solution_present": False}
    assert run.check_egress(good)
    for key, value in [("direct_blocked", False), ("other_host", "HTTP/1.1 200 x"),
                       ("no_auth", "HTTP/1.1 200 x"), ("allowed", "HTTP/1.1 502 x"),
                       ("hidden_tests_present", True)]:
        assert not run.check_egress({**good, key: value})


def ledger_fixture(root):
    ledger = {"summary": "s", "obligations": [{"id": "O001", "content": {"statement": "x"}}],
              "open_questions": [], "preservation_obligations": []}
    relative = "inputs/ledgers/t.json"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    record = {"status": "ledger_ready", "ledger_file": relative,
              "ledger_file_sha256": run.file_sha256(path),
              "ledger_sha256": run.sha256_json(ledger), "ledger_quality": "valid"}
    return ledger, record


def test_solver_input_is_ft_m1_augmentation_pinned_to_the_m0_image(tmp_path):
    ledger, record = ledger_fixture(tmp_path)
    row = {"task_id": "t", "task": {"task_id": "t", "instruction": "Do it.", "image": "tag"},
           "image": {"id": "sha256:1", "repo_digests": ["repo@sha256:2"]}}
    augmented, loaded = run.solver_input(tmp_path, row, record)
    assert loaded == ledger
    assert augmented["image"] == "repo@sha256:2"
    assert augmented["instruction"] == run.ledger_input.augment_task(row["task"], ledger)["instruction"]
    assert augmented["instruction"].count(run.ledger_input.HEADING) == 1
    assert row["task"]["image"] == "tag"
    (tmp_path / record["ledger_file"]).write_text("{}\n")
    with pytest.raises(ValueError):
        run.solver_input(tmp_path, row, record)


def test_review_row_supplies_every_field_review_case_reads(tmp_path):
    _, record = ledger_fixture(tmp_path)
    row = {"task_id": "t", "task": {"task_id": "t"}, "image": {"id": "i", "repo_digests": ["d"]}}
    built = run.review_row(tmp_path, row, record, {"patch_sha256": "p", "reward": 0})
    assert built == {
        "task_id": "t", "task": {"task_id": "t"}, "image": {"id": "i", "repo_digests": ["d"]},
        "source_root": str(tmp_path), "m1_output": str(tmp_path / "solver/t"),
        "ledger_file": str(tmp_path / "inputs/ledgers/t.json"),
        "ledger_sha256": record["ledger_sha256"], "p0_sha256": "p", "m1_reward": 0,
    }
    source = (PROJECT / "scripts/deepswe_ft_m1_review/run.py").read_text()
    body = source[source.index("def review_case("):source.index("def run_batch(")]
    for key in set(built) - {"task_id"}:
        assert f'row["{key}"]' in body, key


def test_partial_attempts_are_never_resampled(tmp_path):
    native = SimpleNamespace(now=lambda: "now")
    (tmp_path / "a/t").mkdir(parents=True)
    (tmp_path / "a/t/partial").write_text("x")
    with pytest.raises(RuntimeError, match="not resampled"):
        run.run_a(native, tmp_path, {"task_id": "t"}, tmp_path / "a/t", "k", {})
    _, record = ledger_fixture(tmp_path)
    (tmp_path / "solver/t").mkdir(parents=True)
    (tmp_path / "solver/t/native.traj.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not resampled"):
        run.run_solver(native, tmp_path, {"task_id": "t"}, record, {})


def case_native():
    return SimpleNamespace(now=lambda: "now", docker=None)


def test_run_case_reuses_terminal_stages_and_tags_the_result(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    _, record = ledger_fixture(tmp_path)
    run.save(tmp_path / "a/t/a-result.json", record)
    run.save(tmp_path / "solver/t/result.json", {"status": "scored", "reward": 0,
                                                 "patch_sha256": "p", "exit_status": "Submitted"})
    monkeypatch.setattr(run, "run_a", lambda *a, **k: pytest.fail("A resampled"))
    monkeypatch.setattr(run, "run_solver", lambda *a, **k: pytest.fail("Solver resampled"))
    monkeypatch.setattr(run, "ensure_image", lambda native, row: None)
    seen = {}

    def review_case(native, root, row):
        seen.update(row)
        return {"status": "scored", "reward": 1, "route": "repair", "p0_reward": row["m1_reward"]}

    monkeypatch.setattr(run.review, "review_case", review_case)
    row = {"task_id": "t", "task": {"task_id": "t"}, "image": {"id": "i", "repo_digests": ["d"]},
           "m0_reward": 0}
    result = run.run_case(case_native(), tmp_path, row, {}, {}, lambda *a, **k: None)
    assert seen["p0_sha256"] == "p" and seen["m1_reward"] == 0
    assert result["reward"] == 1 and result["p0_reward"] == 0 and result["m0_reward"] == 0
    assert result["ledger_quality"] == "valid" and result["experiment_id"] == run.EXPERIMENT_ID


def test_run_case_stops_before_solver_or_review_on_upstream_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setattr(run.review, "review_case", lambda *a, **k: pytest.fail("reviewed"))
    row = {"task_id": "t", "task": {}, "image": {}, "m0_reward": 1}
    monkeypatch.setattr(run, "run_a", lambda *a, **k: {"status": "ledger_unavailable",
                                                       "ledger_error": "schema"})
    monkeypatch.setattr(run, "run_solver", lambda *a, **k: pytest.fail("solved"))
    result = run.run_case(case_native(), tmp_path, row, {}, {}, lambda *a, **k: None)
    assert (result["status"], result["stage"], result["reward"]) == ("ledger_unavailable", "A", None)
    _, record = ledger_fixture(tmp_path)
    monkeypatch.setattr(run, "run_a", lambda *a, **k: record)
    monkeypatch.setattr(run, "run_solver", lambda *a, **k: {
        "status": "infrastructure_or_stage_error", "reward": None, "error": "api"})
    result = run.run_case(case_native(), tmp_path, row, {}, {}, lambda *a, **k: None)
    assert (result["status"], result["stage"]) == ("infrastructure_or_stage_error", "Solver")


def test_a_stage_id_binds_every_input():
    config = run.V62Config(role_timeout_sec=1800, model="deepseek-flash")
    model = run.PiStageModelConfig.from_value(run.STAGE_MODEL_CONFIG)
    base = run.a_stage_id("t", "c", "i", "b", model, config)
    assert base == run.a_stage_id("t", "c", "i", "b", model, config)
    for args in [("u", "c", "i", "b"), ("t", "d", "i", "b"), ("t", "c", "j", "b"), ("t", "c", "i", "e")]:
        assert run.a_stage_id(*args, model, config) != base
    assert config.workflow_controller_version == "v6.2"


def test_runtime_manifest_rejects_a_changed_archive(tmp_path):
    archive = tmp_path / "pi-runtime-a.tar.gz"
    archive.write_bytes(b"archive")
    manifest = {"archive": str(archive), "archive_sha256": run.file_sha256(archive),
                "pi_version": "0.84.2", "node_version": "v22.19.0", "rg_fd_included": False}
    (tmp_path / "runtime.json").write_text(json.dumps(manifest))
    assert run.runtime_manifest(archive)["archive_sha256"] == manifest["archive_sha256"]
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError):
        run.runtime_manifest(archive)
    archive.write_bytes(b"archive")
    (tmp_path / "runtime.json").write_text(json.dumps({**manifest, "rg_fd_included": True}))
    with pytest.raises(ValueError):
        run.runtime_manifest(archive)


def test_excluded_task_is_only_wazero():
    assert set(run.EXCLUDED) == {"wazero-multi-module-snapshots"}
    assert os.path.basename(run.STAGE_MODEL_CONFIG) == "v6.1-deepseek-v4.1-flash.json"


def test_recovery_subset_rejects_unknown_or_excluded_tasks_before_writing(tmp_path):
    m0 = tmp_path / "m0"
    (m0 / "preflight").mkdir(parents=True)
    (m0 / "manifests").mkdir()
    (m0 / "preflight/readiness.json").write_text(json.dumps({"passed": True}))
    (m0 / "manifests/public-tasks.json").write_text(json.dumps(
        [{"task_id": "a"}, {"task_id": "wazero-multi-module-snapshots"}]))
    archive = tmp_path / "pi-runtime-a.tar.gz"
    archive.write_bytes(b"archive")
    (tmp_path / "runtime.json").write_text(json.dumps({
        "archive": str(archive), "archive_sha256": run.file_sha256(archive),
        "pi_version": "0.84.2", "node_version": "v22.19.0", "rg_fd_included": False}))
    for subset in ({"missing"}, {"wazero-multi-module-snapshots"}):
        with pytest.raises(ValueError, match="subset"):
            run.prepare(tmp_path / "root", m0, archive, subset, "recovery")
        assert not (tmp_path / "root").exists()
