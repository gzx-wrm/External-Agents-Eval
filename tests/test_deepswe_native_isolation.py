from __future__ import annotations

import ast
import importlib.util
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/deepswe_native_mini"
if not SCRIPTS.exists():
    SCRIPTS = ROOT / "scripts"  # Remote experiment bundle layout.
HELPER = SCRIPTS / "sanitize_repository.py"


def git(repository, *arguments):
    return subprocess.run(["git", "-C", str(repository), *arguments], check=True,
                          capture_output=True).stdout.decode().strip()


def initialize(path):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Evaluation")
    git(path, "config", "user.email", "eval@localhost")


def helper():
    spec = importlib.util.spec_from_file_location("deepswe_isolation", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initialized_gitlink_keeps_public_files_and_tree_without_history(tmp_path):
    upstream, repository = tmp_path / "upstream", tmp_path / "repository"
    initialize(upstream)
    (upstream / "public.txt").write_text("public fixture\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-qm", "public baseline")
    child_base = git(upstream, "rev-parse", "HEAD")
    (upstream / "future.txt").write_text("unavailable future implementation\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-qm", "future implementation")
    child_future = git(upstream, "rev-parse", "HEAD")
    initialize(repository)
    git(repository, "-c", "protocol.file.allow=always", "submodule", "add", str(upstream), "fixtures")
    git(repository / "fixtures", "checkout", "-q", child_base)
    (repository / "source.py").write_text("value = 1\n")
    git(repository, "add", "-A")
    git(repository, "commit", "-qm", "parent baseline")
    baseline = git(repository, "rev-parse", "HEAD")
    original_tree = git(repository, "rev-parse", "HEAD^{tree}")
    (repository / "future.txt").write_text("unavailable parent implementation\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "parent future")
    parent_future = git(repository, "rev-parse", "HEAD")
    git(repository, "reset", "--hard", baseline)

    audit = helper().sanitize(repository, baseline)

    assert audit["tree_sha"] == original_tree
    assert audit["commit_object_count"] == 1
    assert audit["submodule_snapshots"][0]["git_metadata_removed"] == 1
    assert (repository / "fixtures/public.txt").read_text() == "public fixture\n"
    assert not (repository / "fixtures/.git").exists()
    assert git(repository, "status", "--porcelain") == ""
    git(repository, "clean", "-fd")
    git(repository, "add", "-A")
    assert git(repository, "write-tree") == original_tree
    assert git(repository, "ls-files", "--stage", "fixtures").startswith("160000 " + child_base)
    objects = git(repository, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").splitlines()
    assert parent_future not in objects
    assert child_future not in objects
    assert child_base not in objects
    assert not (repository / ".git/modules").exists()


def test_full_index_diff_matches_after_isolation_despite_abbreviation_change(tmp_path):
    repository = tmp_path / "repository"
    initialize(repository)
    (repository / "source.py").write_text("version = (0, 47, 0)\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "baseline")
    baseline = git(repository, "rev-parse", "HEAD")
    git(repository, "config", "core.abbrev", "12")
    (repository / "source.py").write_text("version = (0, 46, 0)\n")
    source = (SCRIPTS / "run.py").read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "snapshot_command")
    namespace = {"shlex": shlex}
    # Load only the trusted stdlib helper, without requiring Mini on the test host.
    exec(compile(ast.Module(body=[node], type_ignores=[]), "snapshot_command", "exec"), namespace)  # noqa: S102

    def capture(revision):
        return subprocess.check_output(["bash", "-c", namespace["snapshot_command"](revision)],
                                       cwd=repository)

    before = capture(baseline)
    audit = helper().sanitize(repository, baseline)
    subprocess.run(["git", "-C", str(repository), "apply", "--binary", "-"],
                   input=before, check=True, capture_output=True)
    assert capture(audit["synthetic_commit"]) == before
    assert (repository / "source.py").read_text() == "version = (0, 46, 0)\n"
