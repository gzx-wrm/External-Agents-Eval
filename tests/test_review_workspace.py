import asyncio
import copy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents.pier.pi_role_runner import ContainerGitWorkspaceManager
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes


class LocalEnvironment:
    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        result = subprocess.run(  # noqa: ASYNC221 - synchronous fake environment
            ["bash", "-c", command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return SimpleNamespace(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    async def upload_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    async def download_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def git(repository, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_real_git_worktrees_keep_reviewers_isolated_and_publish_p1(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.com")
    source = repository / "client.py"
    source.write_text("timeout = None\n")
    git(repository, "add", "client.py")
    git(repository, "commit", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").strip()
    source.write_text("timeout = 10\n")
    p0 = git(repository, "diff", "--binary", "--no-ext-diff", base, "--", ".").encode()
    git(repository, "restore", "client.py")

    manager = ContainerGitWorkspaceManager(
        environment=LocalEnvironment(),
        transfer_dir=tmp_path / "transfer",
        repository=str(repository),
    )
    manager.ROOT = str(tmp_path / "worktrees")
    manager.INPUT_ROOT = str(tmp_path / "contracts")

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)
        frozen_path = await manager.materialize_file(
            name="frozen.txt", payload=b"frozen\n", readonly=True
        )
        assert await manager.materialized_file_digest(path=frozen_path) == sha256_bytes(b"frozen\n")
        assert Path(frozen_path).stat().st_mode & 0o222 == 0
        await manager.apply_patch(worktree=worktrees.patch_review, patch=p0)
        assert await manager.tracked_diff_digest(
            worktree=worktrees.patch_review, base_commit=base
        ) == sha256_bytes(p0)
        Path(worktrees.spec_review, "client.py").write_text("reviewer mutation\n")
        assert await manager.tracked_diff_digest(
            worktree=worktrees.spec_review, base_commit=base
        ) != sha256_bytes(b"")
        errors = await manager.validate_evidence(
            worktree=worktrees.patch_review,
            assessment={
                "obligation_assessments": [
                    {
                        "obligation_id": "O01",
                        "implementation_evidence": [{"path": "client.py", "line_end": 1}],
                        "test_evidence": [],
                    }
                ]
            },
        )
        assert errors == []
        await manager.publish_final_patch(base_commit=base, patch=p0)
        await manager.cleanup()

    asyncio.run(exercise())
    assert source.read_text() == "timeout = 10\n"
    assert not (tmp_path / "worktrees" / "spec-review").exists()


def make_manager(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.com")
    (repository / "client.py").write_text("timeout = None\n")
    git(repository, "add", "client.py")
    git(repository, "commit", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").strip()
    manager = ContainerGitWorkspaceManager(
        environment=LocalEnvironment(),
        transfer_dir=tmp_path / "transfer",
        repository=str(repository),
    )
    manager.ROOT = str(tmp_path / "worktrees")
    manager.INPUT_ROOT = str(tmp_path / "contracts")
    return repository, base, manager


def test_empty_patch_is_a_noop(tmp_path):
    repository, base, manager = make_manager(tmp_path)

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)
        await manager.apply_patch(worktree=worktrees.patch_review, patch=b"")
        assert (
            await manager.capture_patch(
                worktree=worktrees.patch_review,
                base_commit=base,
            )
            == b""
        )
        await manager.cleanup()

    asyncio.run(exercise())
    assert git(repository, "status", "--porcelain") == ""


def test_v2_late_and_regression_evidence_paths_are_validated(tmp_path):
    _, base, manager = make_manager(tmp_path)

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)
        implementation = {
            "path": "missing-implementation.py",
            "symbol": None,
            "line_start": 1,
            "line_end": 1,
            "explanation": "The missing path must be rejected.",
        }
        test = {
            "path": "missing-test.py",
            "test_name": "test_missing",
            "command": "python -m pytest missing-test.py",
            "result": "failed",
            "explanation": "The missing test path must be rejected.",
        }
        errors = await manager.validate_evidence(
            worktree=worktrees.patch_review,
            assessment={
                "criterion_assessments": [],
                "late_obligation_candidates": [
                    {
                        "criterion_ref": "L01.C01",
                        "implementation_evidence": [implementation],
                        "test_evidence": [test],
                    }
                ],
                "unexpected_regressions": [
                    {
                        "id": "R01",
                        "implementation_evidence": [implementation],
                    }
                ],
            },
        )
        await manager.cleanup()
        return errors

    errors = asyncio.run(exercise())
    assert any("L01.C01 evidence path does not exist: missing-implementation.py" in item for item in errors)
    assert any("L01.C01 evidence path does not exist: missing-test.py" in item for item in errors)
    assert any("R01 evidence path does not exist: missing-implementation.py" in item for item in errors)


def test_v2_repository_sources_are_validated_against_frozen_base(tmp_path):
    repository, base, manager = make_manager(tmp_path)
    content = (repository / "client.py").read_bytes()

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)
        source = {
            "id": "S01",
            "kind": "repository",
            "locator": {
                "path": "client.py",
                "base_revision": base,
                "line_start": 1,
                "line_end": 1,
                "quote": "timeout = None",
                "file_sha256": sha256_bytes(content),
            },
        }
        assert (
            await manager.validate_repository_sources(
                worktree=worktrees.spec_review,
                base_commit=base,
                sources=[source],
            )
            == []
        )
        source["locator"]["quote"] = "timeout = 10"
        errors = await manager.validate_repository_sources(
            worktree=worktrees.spec_review,
            base_commit=base,
            sources=[source],
        )
        assert errors == ["S01 repository source quote does not match: client.py"]

        metadata = await manager.materialize_file(
            name="repair_audit.json", payload=b'{"valid":true}'
        )
        assert await manager.read_materialized_file(path=metadata) == b'{"valid":true}'
        assert not (tmp_path / "transfer/read-repair_audit.json").exists()
        await manager.cleanup()

    asyncio.run(exercise())


def test_v2_repository_source_validation_rejects_wrong_base_metadata_and_candidate_files(
    tmp_path,
):
    repository, _, manager = make_manager(tmp_path)
    (repository / "client-link.py").symlink_to("client.py")
    git(repository, "add", "client-link.py")
    git(repository, "commit", "-m", "add public symlink")
    base = git(repository, "rev-parse", "HEAD").strip()
    content = (repository / "client.py").read_bytes()
    valid = {
        "id": "S01",
        "kind": "repository",
        "locator": {
            "path": "client.py",
            "base_revision": base,
            "line_start": 1,
            "line_end": 1,
            "quote": "timeout = None",
            "file_sha256": sha256_bytes(content),
        },
    }

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)

        wrong_digest = copy.deepcopy(valid)
        wrong_digest["locator"]["file_sha256"] = "0" * 64
        assert (
            "digest does not match"
            in (
                await manager.validate_repository_sources(
                    worktree=worktrees.spec_review,
                    base_commit=base,
                    sources=[wrong_digest],
                )
            )[0]
        )

        wrong_range = copy.deepcopy(valid)
        wrong_range["locator"]["line_end"] = 2
        assert (
            "line range is invalid"
            in (
                await manager.validate_repository_sources(
                    worktree=worktrees.spec_review,
                    base_commit=base,
                    sources=[wrong_range],
                )
            )[0]
        )

        wrong_revision = copy.deepcopy(valid)
        wrong_revision["locator"]["base_revision"] = "f" * 40
        assert (
            "base revision does not match"
            in (
                await manager.validate_repository_sources(
                    worktree=worktrees.spec_review,
                    base_commit=base,
                    sources=[wrong_revision],
                )
            )[0]
        )

        impossible_long_revision = copy.deepcopy(valid)
        impossible_long_revision["locator"]["base_revision"] = base + "f" * 24
        assert (
            "base revision does not match"
            in (
                await manager.validate_repository_sources(
                    worktree=worktrees.spec_review,
                    base_commit=base,
                    sources=[impossible_long_revision],
                )
            )[0]
        )

        symlink = copy.deepcopy(valid)
        symlink["locator"]["path"] = "client-link.py"
        assert (
            "not a regular file"
            in (
                await manager.validate_repository_sources(
                    worktree=worktrees.spec_review,
                    base_commit=base,
                    sources=[symlink],
                )
            )[0]
        )

        (Path(worktrees.patch_review) / "candidate_only.py").write_text("candidate = True\n")
        candidate = copy.deepcopy(valid)
        candidate["locator"].update(
            {
                "path": "candidate_only.py",
                "quote": "candidate = True",
                "file_sha256": sha256_bytes(b"candidate = True\n"),
            }
        )
        assert (
            "does not exist at base"
            in (
                await manager.validate_repository_sources(
                    worktree=worktrees.patch_review,
                    base_commit=base,
                    sources=[candidate],
                )
            )[0]
        )
        await manager.cleanup()

    asyncio.run(exercise())


def test_v5_code_references_are_validated_against_the_named_snapshot(tmp_path):
    repository, _, manager = make_manager(tmp_path)
    (repository / "base-link.py").symlink_to("client.py")
    git(repository, "add", "base-link.py")
    git(repository, "commit", "-m", "add base symlink")
    base = git(repository, "rev-parse", "HEAD").strip()

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)
        candidate = Path(worktrees.patch_review)
        (candidate / "candidate_only.py").write_text("first = 1\nsecond = 2\n")
        (candidate / "candidate-link.py").symlink_to("candidate_only.py")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "escaped.py").write_text("escaped = True\n")
        (candidate / "outside-link").symlink_to(outside, target_is_directory=True)

        valid = {
            "target_id": "O001",
            "code_refs": [
                {
                    "snapshot": "base",
                    "path": "client.py",
                    "line_start": 1,
                    "line_end": 1,
                },
                {
                    "snapshot": "candidate",
                    "path": "candidate_only.py",
                    "line_start": 2,
                    "line_end": 2,
                },
            ],
        }
        assert (
            await manager.validate_v5_code_references(
                base_worktree=worktrees.spec_review,
                candidate_worktree=worktrees.patch_review,
                base_commit=base,
                assessments=[valid],
            )
            == []
        )

        invalid = {
            "target_id": "O002",
            "code_refs": [
                {
                    "snapshot": "base",
                    "path": "../client.py",
                    "line_start": 1,
                    "line_end": 1,
                },
                {
                    "snapshot": "base",
                    "path": "missing.py",
                    "line_start": 1,
                    "line_end": 1,
                },
                {
                    "snapshot": "base",
                    "path": "base-link.py",
                    "line_start": 1,
                    "line_end": 1,
                },
                {
                    "snapshot": "base",
                    "path": "client.py",
                    "line_start": 1,
                    "line_end": 2,
                },
                {
                    "snapshot": "candidate",
                    "path": "missing-candidate.py",
                    "line_start": 1,
                    "line_end": 1,
                },
                {
                    "snapshot": "candidate",
                    "path": "candidate-link.py",
                    "line_start": 1,
                    "line_end": 1,
                },
                {
                    "snapshot": "candidate",
                    "path": "candidate_only.py",
                    "line_start": 1,
                    "line_end": 3,
                },
                {
                    "snapshot": "candidate",
                    "path": "outside-link/escaped.py",
                    "line_start": 1,
                    "line_end": 1,
                },
            ],
        }
        errors = await manager.validate_v5_code_references(
            base_worktree=worktrees.spec_review,
            candidate_worktree=worktrees.patch_review,
            base_commit=base,
            assessments=[invalid],
        )
        await manager.cleanup()
        return errors

    errors = asyncio.run(exercise())
    assert any("unsafe path: ../client.py" in item for item in errors)
    assert any("base code reference is missing: missing.py" in item for item in errors)
    assert any("base code reference is not a regular file: base-link.py" in item for item in errors)
    assert any("base code reference range is invalid: client.py" in item for item in errors)
    assert any("candidate code reference range is invalid: missing-candidate.py" in item for item in errors)
    assert any("candidate code reference range is invalid: candidate-link.py" in item for item in errors)
    assert any("candidate code reference range is invalid: candidate_only.py" in item for item in errors)
    assert any(
        "candidate code reference range is invalid: outside-link/escaped.py" in item
        for item in errors
    )


def test_capture_patch_includes_untracked_files_without_mutating_real_index(tmp_path):
    _, base, manager = make_manager(tmp_path)

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)
        patch_worktree = Path(worktrees.patch_review)
        (patch_worktree / "client.py").write_text("timeout = 10\n")
        git(patch_worktree, "add", "client.py")
        (patch_worktree / "new_test.py").write_text("def test_timeout():\n    pass\n")
        index_before = git(patch_worktree, "diff", "--cached", "--binary")
        status_before = git(patch_worktree, "status", "--porcelain")

        patch = await manager.capture_patch(
            worktree=worktrees.patch_review,
            base_commit=base,
        )

        assert b"new_test.py" in patch
        assert b"client.py" in patch
        assert await manager.tracked_diff_digest(
            worktree=worktrees.patch_review,
            base_commit=base,
        ) == sha256_bytes(patch)
        assert git(patch_worktree, "diff", "--cached", "--binary") == index_before
        assert git(patch_worktree, "status", "--porcelain") == status_before
        await manager.cleanup()

    asyncio.run(exercise())


def test_capture_patch_can_be_relative_to_environment_baseline(tmp_path):
    repository, base, manager = make_manager(tmp_path)
    source = repository / "client.py"
    source.write_text("environment baseline\n")

    async def exercise():
        baseline = await manager.capture_patch(
            worktree=str(repository),
            base_commit=base,
        )
        await manager.restore_final_worktree(base_commit=base)
        worktrees = await manager.prepare(base_commit=base)
        await manager.apply_patch(worktree=worktrees.patch_review, patch=baseline)
        Path(worktrees.patch_review, "client.py").write_text("candidate change\n")

        relative = await manager.capture_patch(
            worktree=worktrees.patch_review,
            base_commit=base,
            baseline_patch=baseline,
        )

        assert b"candidate change" in relative
        assert b"+environment baseline" not in relative
        assert await manager.tracked_diff_digest(
            worktree=worktrees.patch_review,
            base_commit=base,
            baseline_patch=baseline,
        ) == sha256_bytes(relative)
        await manager.cleanup()

    asyncio.run(exercise())


def test_restore_final_worktree_discards_tracked_and_untracked_pollution(tmp_path):
    repository, base, manager = make_manager(tmp_path)

    async def exercise():
        await manager.prepare(base_commit=base)
        (repository / "client.py").write_text("polluted\n")
        (repository / "untracked.txt").write_text("polluted\n")

        discarded = await manager.restore_final_worktree(base_commit=base)

        assert b"client.py" in discarded
        assert b"untracked.txt" in discarded
        assert (repository / "client.py").read_text() == "timeout = None\n"
        assert not (repository / "untracked.txt").exists()
        assert git(repository, "status", "--porcelain") == ""
        await manager.cleanup()

    asyncio.run(exercise())


def test_spec_reviewer_restore_allows_only_untracked_scratch(tmp_path):
    repository, base, manager = make_manager(tmp_path)

    async def exercise():
        worktrees = await manager.prepare(base_commit=base)
        spec_worktree = Path(worktrees.spec_review)
        scratch = spec_worktree / "issue_body.txt"
        scratch.write_text("temporary issue copy\n")

        assert await manager.tracked_files_diff_digest(
            worktree=worktrees.spec_review,
            base_commit=base,
        ) == sha256_bytes(b"")
        assert await manager.tracked_diff_digest(
            worktree=worktrees.spec_review,
            base_commit=base,
        ) != sha256_bytes(b"")

        discarded = await manager.restore_reviewer_worktree(
            worktree=worktrees.spec_review,
            base_commit=base,
        )
        assert b"issue_body.txt" in discarded
        assert not scratch.exists()
        assert git(spec_worktree, "status", "--porcelain") == ""

        (spec_worktree / "client.py").write_text("timeout = 10\n")
        assert await manager.tracked_files_diff_digest(
            worktree=worktrees.spec_review,
            base_commit=base,
        ) != sha256_bytes(b"")
        await manager.cleanup()

    asyncio.run(exercise())
    assert git(repository, "status", "--porcelain") == ""


def test_publish_final_patch_still_rejects_dirty_app(tmp_path):
    repository, base, manager = make_manager(tmp_path)

    async def exercise():
        await manager.prepare(base_commit=base)
        (repository / "client.py").write_text("polluted\n")
        with pytest.raises(RuntimeError, match="modified /app"):
            await manager.publish_final_patch(base_commit=base, patch=b"")
        await manager.cleanup()

    asyncio.run(exercise())


def test_publish_final_patch_preserves_recorded_environment_baseline(tmp_path):
    repository, base, manager = make_manager(tmp_path)
    source = repository / "client.py"
    source.write_text("environment baseline\n")

    async def exercise():
        baseline = await manager.capture_patch(worktree=str(repository), base_commit=base)
        await manager.prepare(base_commit=base)
        await manager.restore_final_worktree(base_commit=base)
        await manager.apply_patch(worktree=str(repository), patch=baseline)
        await manager.publish_final_patch(
            base_commit=base,
            patch=b"",
            baseline_patch=baseline,
        )
        source.write_text("unexpected reviewer mutation\n")
        with pytest.raises(RuntimeError, match="outside the recorded baseline"):
            await manager.publish_final_patch(
                base_commit=base,
                patch=b"",
                baseline_patch=baseline,
            )
        await manager.cleanup()

    asyncio.run(exercise())


def test_publish_baseline_relative_patch_over_environment_state(tmp_path):
    repository, base, manager = make_manager(tmp_path)
    source = repository / "client.py"
    source.write_text("environment baseline\n")

    async def exercise():
        baseline = await manager.capture_patch(
            worktree=str(repository),
            base_commit=base,
        )
        await manager.restore_final_worktree(base_commit=base)
        await manager.prepare(base_commit=base)
        await manager.apply_patch(worktree=str(repository), patch=baseline)
        source.write_text("candidate change\n")
        candidate = await manager.capture_patch(
            worktree=str(repository),
            base_commit=base,
            baseline_patch=baseline,
        )
        await manager.restore_final_worktree(base_commit=base)
        await manager.apply_patch(worktree=str(repository), patch=baseline)
        await manager.publish_final_patch(
            base_commit=base,
            patch=candidate,
            baseline_patch=baseline,
        )
        assert source.read_text() == "candidate change\n"
        await manager.cleanup()

    asyncio.run(exercise())
