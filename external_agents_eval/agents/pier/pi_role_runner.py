from __future__ import annotations

import asyncio
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pier.environments.base import BaseEnvironment
from pier.utils.trajectory_utils import format_trajectory_json

from external_agents_eval.agents.pier.pi import _build_pi_command
from external_agents_eval.agents.pier.trajectory import (
    extract_pi_final_assistant_text,
    pi_jsonl_to_atif,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import (
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    IndeterminateInvocationError,
    RoleName,
    RoleRunResult,
    WorkflowWorktrees,
)


def _temporary_index_diff_command(base_commit: str) -> str:
    """Capture tracked and untracked changes without mutating the real Git index."""
    quoted_base = shlex.quote(base_commit)
    return (
        'snapshot_dir="$(mktemp -d)" && '
        "trap 'rm -rf -- \"$snapshot_dir\"' EXIT && "
        'snapshot_index="$snapshot_dir/index" && '
        f'GIT_INDEX_FILE="$snapshot_index" git read-tree {quoted_base} && '
        'GIT_INDEX_FILE="$snapshot_index" git add -A -- . && '
        f'GIT_INDEX_FILE="$snapshot_index" git diff --cached --binary '
        f"--no-ext-diff {quoted_base} -- ."
    )


def _temporary_index_tracked_diff_command(base_commit: str) -> str:
    """Capture changes to base-tracked paths while ignoring new scratch files."""
    quoted_base = shlex.quote(base_commit)
    return (
        'snapshot_dir="$(mktemp -d)" && '
        "trap 'rm -rf -- \"$snapshot_dir\"' EXIT && "
        'snapshot_index="$snapshot_dir/index" && '
        f'GIT_INDEX_FILE="$snapshot_index" git read-tree {quoted_base} && '
        'GIT_INDEX_FILE="$snapshot_index" git add -u -- . && '
        f'GIT_INDEX_FILE="$snapshot_index" git diff --cached --binary '
        f"--no-ext-diff {quoted_base} -- ."
    )


class PiRoleRunner:
    """Run isolated Pi processes without implicitly committing repository changes."""

    REVIEW_OUTPUT_ROOT = "/tmp/external_eval/v5_review_notes"

    def __init__(
        self,
        *,
        environment: BaseEnvironment,
        logs_root: Path,
        model: str,
        agent_version: str,
        env: dict[str, str],
        redact,
        invocation_recovery_wait_sec: float = 5.0,
    ) -> None:
        self.environment = environment
        self.logs_root = logs_root
        self.model = model
        self.agent_version = agent_version
        self.env = env
        self.redact = redact
        # A transport timeout does not tell us whether the remote shell was
        # stopped.  On a subsequent controller process we wait briefly for a
        # completion record, then leave the invocation recoverable instead of
        # starting a second Pi process with the same identity.
        self.invocation_recovery_wait_sec = max(0.0, float(invocation_recovery_wait_sec))
        # The first invocation creates a timestamped session file.  Keep the
        # resolved path in memory so later invocations use ``--session`` with
        # that exact file instead of asking Pi to rediscover it by ID.
        self._session_paths: dict[str, str] = {}

    async def prepare_review_note_skill(self, source_root: Path) -> str:
        """Materialize the versioned review-note skill inside the task container."""
        source_root = source_root.resolve()
        files = sorted(path for path in source_root.rglob("*") if path.is_file())
        if not files or not (source_root / "SKILL.md").is_file():
            raise FileNotFoundError(f"Invalid review-note skill package: {source_root}")
        digest_payload = b"".join(
            path.relative_to(source_root).as_posix().encode("utf-8")
            + b"\0"
            + path.read_bytes()
            + b"\0"
            for path in files
        )
        digest = sha256_bytes(digest_payload)
        remote_root = f"/external_eval/v5_resources/v5-review-note-{digest[:16]}"
        complete = await self.environment.exec(
            f"test -f {shlex.quote(remote_root + '/.complete')}"
        )
        if complete.return_code == 0:
            return f"{remote_root}/SKILL.md"
        await self.environment.exec(
            f"chmod -R u+w {shlex.quote(remote_root)} 2>/dev/null || true; "
            f"rm -rf -- {shlex.quote(remote_root)}; mkdir -p {shlex.quote(remote_root)}"
        )
        for source in files:
            relative = source.relative_to(source_root).as_posix()
            destination = f"{remote_root}/{relative}"
            await self.environment.exec(
                f"mkdir -p {shlex.quote(str(Path(destination).parent))}"
            )
            await self.environment.upload_file(source, destination)
        await self.environment.exec(
            f"touch {shlex.quote(remote_root + '/.complete')} && "
            f"chmod -R a-w {shlex.quote(remote_root)}"
        )
        return f"{remote_root}/SKILL.md"

    async def prepare_review_catalog(
        self, *, role: str, catalog: dict[str, Any]
    ) -> str:
        """Publish one immutable, content-addressed target catalog."""
        payload = (
            json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        digest = sha256_bytes(payload)
        if role not in {"I", "R"}:
            raise ValueError(f"Invalid V5 catalog role: {role}")
        remote = f"/external_eval/v5_resources/catalog-{role}-{digest[:16]}.json"
        exists = await self.environment.exec(f"test -f {shlex.quote(remote)}")
        if exists.return_code == 0:
            return remote
        host = self.logs_root / "v5" / "catalogs" / f"{digest}.json"
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_bytes(payload)
        await self.environment.exec("mkdir -p /external_eval/v5_resources")
        await self.environment.upload_file(host, remote)
        await self.environment.exec(f"chmod 0444 {shlex.quote(remote)}")
        return remote

    async def prepare_review_output_root(self, artifact_key: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", artifact_key):
            raise ValueError(f"Unsafe V5 output key: {artifact_key}")
        remote_root = f"{self.REVIEW_OUTPUT_ROOT}/{artifact_key}"
        exists = await self.environment.exec(f"test -e {shlex.quote(remote_root)}")
        if exists.return_code == 0:
            raise RuntimeError(f"Refusing to reuse V5 Reviewer output root {remote_root}")
        created = await self.environment.exec(f"mkdir -p {shlex.quote(remote_root)}")
        if created.return_code != 0:
            raise RuntimeError(f"Could not create V5 Reviewer output root {remote_root}")
        return remote_root

    async def resume_review_output_root(self, output_root: str) -> str:
        """Validate a PREPARED attempt's existing scratch root for recovery."""
        if not output_root.startswith(self.REVIEW_OUTPUT_ROOT + "/"):
            raise ValueError(f"Unsafe V5 Reviewer output root: {output_root}")
        check = await self.environment.exec(
            f"parent_real=$(realpath -- {shlex.quote(self.REVIEW_OUTPUT_ROOT)}) && "
            f"root_real=$(realpath -- {shlex.quote(output_root)}) && "
            'case "$root_real" in "$parent_real"/*) ;; *) exit 1;; esac && '
            f"test -d {shlex.quote(output_root)} && "
            # The output root is disposable scratch.  Symlinks inside it are
            # allowed so reviewers can reuse dependency trees during a
            # resumed attempt; declared artifacts are checked independently.
            f"test ! -L {shlex.quote(output_root)}"
        )
        if check.return_code != 0:
            raise IndeterminateInvocationError(
                f"Prepared V5 Reviewer output root is unavailable: {output_root}"
            )
        return output_root

    async def download_review_file(
        self,
        *,
        output_root: str,
        relative_path: str,
        host_path: Path,
    ) -> bytes:
        """Download one declared Reviewer file with path-local containment checks.

        Reviewer scratch may contain symlinks (for example a dependency cache),
        but a declared note must itself be a regular file and its resolved path
        must remain below the attempt root.  This keeps the artifact boundary
        strict without treating unrelated scratch entries as protocol failures.
        """
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe V5 Reviewer artifact path: {relative_path}")
        if not output_root.startswith(self.REVIEW_OUTPUT_ROOT + "/"):
            raise ValueError(f"Unsafe V5 Reviewer output root: {output_root}")
        remote_path = f"{output_root}/{relative.as_posix()}"
        if host_path.is_symlink() or ".." in host_path.parts:
            raise ValueError(f"Unsafe host V5 Reviewer artifact path: {host_path}")
        check = await self.environment.exec(
            f"parent_real=$(realpath -- {shlex.quote(self.REVIEW_OUTPUT_ROOT)}) && "
            "root_real=$(realpath -- "
            f"{shlex.quote(output_root)}) && "
            'case "$root_real" in "$parent_real"/*) ;; *) exit 1;; esac && '
            "candidate_real=$(realpath -- "
            f"{shlex.quote(remote_path)}) && "
            f"test -f {shlex.quote(remote_path)} && "
            f"test ! -L {shlex.quote(remote_path)} && "
            'case "$candidate_real" in "$root_real"/*) exit 0;; *) exit 1;; esac'
        )
        if check.return_code != 0:
            raise RuntimeError(f"Unsafe or missing V5 Reviewer artifact: {relative_path}")
        remote_digest_before = await self._remote_sha256(remote_path)
        if remote_digest_before is None:
            raise RuntimeError(f"Could not hash V5 Reviewer artifact: {relative_path}")
        host_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = host_path.with_name(host_path.name + ".partial")
        temporary.unlink(missing_ok=True)
        await self.environment.download_file(remote_path, temporary)
        content = temporary.read_bytes()
        remote_digest_after = await self._remote_sha256(remote_path)
        if remote_digest_after != remote_digest_before or sha256_bytes(content) != remote_digest_before:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"V5 Reviewer artifact changed during archive: {relative_path}")
        temporary.replace(host_path)
        return content

    async def download_review_materials(
        self,
        *,
        output_root: str,
        host_root: Path,
        max_files: int = 128,
        max_file_bytes: int = 16 * 1024 * 1024,
        max_total_bytes: int = 32 * 1024 * 1024,
    ) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
        """Archive bounded reviewer-authored probe and log files."""
        if not output_root.startswith(self.REVIEW_OUTPUT_ROOT + "/"):
            raise ValueError(f"Unsafe V5 Reviewer output root: {output_root}")
        root_check = await self.environment.exec(
            f"parent_real=$(realpath -- {shlex.quote(self.REVIEW_OUTPUT_ROOT)}) && "
            f"root_real=$(realpath -- {shlex.quote(output_root)}) && "
            'case "$root_real" in "$parent_real"/*) ;; *) exit 1;; esac && '
            f"test -d {shlex.quote(output_root)} && test ! -L {shlex.quote(output_root)}"
        )
        if root_check.return_code != 0:
            raise ValueError(f"Unsafe V5 Reviewer output root: {output_root}")
        listing = await self.environment.exec(
            f"if test -d {shlex.quote(output_root + '/probes')}; then "
            f"find {shlex.quote(output_root + '/probes')} -type f -printf "
            "'probes/%P\\n'; fi; "
            f"if test -d {shlex.quote(output_root + '/logs')}; then "
            f"find {shlex.quote(output_root + '/logs')} -type f -printf "
            "'logs/%P\\n'; fi"
        )
        if listing.return_code != 0:
            raise RuntimeError("Could not enumerate V5 Reviewer materials")
        relative_paths = sorted(
            {line for line in (listing.stdout or "").splitlines() if line}
        )
        if len(relative_paths) > max_files:
            raise RuntimeError(
                f"V5 Reviewer material count exceeds limit {max_files}"
            )
        files: dict[str, bytes] = {}
        manifest: list[dict[str, Any]] = []
        total = 0
        for relative_path in relative_paths:
            relative = Path(relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[:1] not in {("probes",), ("logs",)}
                or any(character in relative_path for character in ("\0", "\r", "\n"))
            ):
                raise RuntimeError(
                    f"Unsafe V5 Reviewer material path: {relative_path}"
                )
            remote_path = f"{output_root}/{relative.as_posix()}"
            size_result = await self.environment.exec(
                f"test -f {shlex.quote(remote_path)} && "
                f"test ! -L {shlex.quote(remote_path)} && "
                f"root_real=$(realpath -- {shlex.quote(output_root)}) && "
                f"candidate_real=$(realpath -- {shlex.quote(remote_path)}) && "
                f'case "$candidate_real" in "$root_real"/*) ;; *) exit 1;; esac && '
                f"wc -c < {shlex.quote(remote_path)}"
            )
            try:
                size = int((size_result.stdout or "").strip())
            except ValueError:
                size = -1
            if size_result.return_code != 0 or size < 0:
                raise RuntimeError(
                    f"Could not validate V5 Reviewer material: {relative_path}"
                )
            digest_before = await self._remote_sha256(remote_path)
            if digest_before is None:
                raise RuntimeError(f"Could not hash V5 Reviewer material: {relative_path}")
            if size > max_file_bytes or total + size > max_total_bytes:
                raise RuntimeError(
                    f"V5 Reviewer material size exceeds archive limit: {relative_path}"
                )
            host_path = host_root / relative
            host_path.parent.mkdir(parents=True, exist_ok=True)
            if host_path.is_symlink() or ".." in host_path.parts:
                raise RuntimeError(f"Unsafe host V5 Reviewer material path: {relative_path}")
            temporary = host_path.with_name(host_path.name + ".partial")
            temporary.unlink(missing_ok=True)
            await self.environment.download_file(remote_path, temporary)
            content = temporary.read_bytes()
            digest_after = await self._remote_sha256(remote_path)
            if (
                len(content) != size
                or digest_before != digest_after
                or sha256_bytes(content) != digest_before
            ):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"V5 Reviewer material changed during archive: {relative_path}"
                )
            temporary.replace(host_path)
            archived_path = f"materials/{relative.as_posix()}"
            files[archived_path] = content
            manifest.append(
                {
                    "original_path": remote_path,
                    "archived_path": archived_path,
                    "sha256": sha256_bytes(content),
                    "material_type": relative.parts[0][:-1],
                    "version_evidence": "captured_after_reviewer_process_exit",
                }
            )
            total += size
        return files, manifest

    async def cleanup_review_output_root(self, output_root: str) -> None:
        if not output_root.startswith(self.REVIEW_OUTPUT_ROOT + "/"):
            raise ValueError(f"Unsafe V5 Reviewer output root: {output_root}")
        await self.environment.exec(f"rm -rf -- {shlex.quote(output_root)}")

    async def materialize_evidence_bundles(
        self, artifacts: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Copy immutable host bundles into one read-only container evidence tree."""
        local_bundle_root = (self.logs_root / "v5" / "bundles").resolve()
        remote_root = "/external_eval/v5_evidence"
        rewritten: list[dict[str, Any]] = []
        for artifact in artifacts:
            digest = str(artifact["bundle_sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid V5 bundle digest: {digest}")
            source_root = Path(artifact["bundle_path"]).resolve()
            if source_root.parent != local_bundle_root or source_root.name != digest:
                raise ValueError(f"V5 bundle is outside the artifact store: {source_root}")
            self._verify_bundle(source_root, digest)
            remote_bundle = f"{remote_root}/{digest}"
            complete = await self.environment.exec(
                f"test -f {shlex.quote(remote_bundle + '/.complete')}"
            )
            if complete.return_code != 0:
                await self.environment.exec(
                    f"chmod -R u+w {shlex.quote(remote_bundle)} 2>/dev/null || true; "
                    f"rm -rf -- {shlex.quote(remote_bundle)}; "
                    f"mkdir -p {shlex.quote(remote_bundle)}"
                )
                for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
                    relative = source.relative_to(source_root).as_posix()
                    destination = f"{remote_bundle}/{relative}"
                    await self.environment.exec(
                        f"mkdir -p {shlex.quote(str(Path(destination).parent))}"
                    )
                    await self.environment.upload_file(source, destination)
                await self.environment.exec(
                    f"touch {shlex.quote(remote_bundle + '/.complete')}"
                )
            updated = dict(artifact)
            updated["bundle_path"] = remote_bundle
            for key in ("note_path", "trace_manifest_path", "materials_manifest_path"):
                local_path = Path(str(artifact[key])).resolve()
                try:
                    relative = local_path.relative_to(source_root).as_posix()
                except ValueError as error:
                    raise ValueError(f"V5 artifact path escapes its bundle: {local_path}") from error
                updated[key] = f"{remote_bundle}/{relative}"
            rewritten.append(updated)
        await self.environment.exec(f"chmod -R a-w {shlex.quote(remote_root)}")
        return remote_root, rewritten

    @staticmethod
    def _verify_bundle(source_root: Path, expected_digest: str) -> None:
        manifest_path = source_root / "bundle-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_json(manifest) != expected_digest:
            raise ValueError(f"V5 bundle manifest digest mismatch: {expected_digest}")
        hashes = {
            **manifest.get("files", {}),
            **manifest.get("note_hashes", {}),
            **manifest.get("extra_hashes", {}),
        }
        for relative, expected in hashes.items():
            path = source_root / relative
            if not path.is_file() or path.is_symlink() or sha256_bytes(path.read_bytes()) != expected:
                raise ValueError(f"V5 bundle file failed integrity check: {relative}")

    async def run_role(
        self,
        *,
        role: RoleName,
        prompt: str,
        worktree: str,
        tools: tuple[str, ...],
        timeout_sec: float | None,
        artifact_prefix: str,
        session_id: str | None = None,
        session_dir: str | None = None,
        session_file: str | None = None,
        no_compaction: bool = False,
        invocation_id: str | None = None,
    ) -> RoleRunResult:
        started = datetime.now(timezone.utc)
        host_dir = self.logs_root / artifact_prefix
        host_dir.mkdir(parents=True, exist_ok=True)
        prompt_host = host_dir / "prompt.md"
        prompt_host.write_text(self.redact(prompt), encoding="utf-8")
        safe_prefix = artifact_prefix.replace("/", "__")
        invocation_replayed = False
        invocation_state = "NEW"
        if invocation_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", invocation_id):
                raise ValueError(f"Unsafe invocation ID: {invocation_id}")
            remote_dir = f"/external_eval/v5_invocations/{invocation_id}"
            await self.environment.exec("mkdir -p /external_eval/v5_invocations")
            created = await self.environment.exec(
                f"mkdir {shlex.quote(remote_dir)}"
            )
            if created.return_code != 0:
                completed = await self.environment.exec(
                    f"test -f {shlex.quote(remote_dir + '/completion.json')}"
                )
                if completed.return_code != 0:
                    # A controller restart can race the remote launcher.  Do
                    # not infer failure from the missing completion file and
                    # do not launch a duplicate process.  The launcher state
                    # and heartbeat are advisory; the completion record is
                    # still the only accepted result proof.
                    await self._wait_for_invocation_completion(
                        remote_dir=remote_dir,
                        invocation_id=invocation_id,
                    )
                    completed = await self.environment.exec(
                        f"test -f {shlex.quote(remote_dir + '/completion.json')}"
                    )
                if completed.return_code != 0:
                    state = await self._read_invocation_state(remote_dir)
                    raise IndeterminateInvocationError(
                        f"Invocation {invocation_id} exists without a trusted completion "
                        f"record (state={state})"
                    )
                request = await self.environment.exec(
                    f"cat {shlex.quote(remote_dir + '/invocation.json')}"
                )
                try:
                    request_record = json.loads(request.stdout or "{}")
                except json.JSONDecodeError as error:
                    raise IndeterminateInvocationError(
                        f"Invocation {invocation_id} has an invalid request record"
                    ) from error
                expected_request = {
                    "invocation_id": invocation_id,
                    "role": role.value,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "worktree": worktree,
                }
                if request.return_code != 0 or any(
                    request_record.get(key) != value
                    for key, value in expected_request.items()
                ):
                    raise IndeterminateInvocationError(
                        f"Invocation {invocation_id} is bound to different execution inputs"
                    )
                invocation_replayed = True
                invocation_state = "COMPLETED"
        else:
            remote_dir = f"/tmp/external_eval/role_runs/{safe_prefix}"
        prompt_remote = f"{remote_dir}/prompt.md"
        transcript_remote = f"{remote_dir}/pi.jsonl"
        stderr_remote = f"{remote_dir}/stderr.txt"
        if not invocation_replayed:
            await self.environment.exec(f"mkdir -p {shlex.quote(remote_dir)}")
            await self.environment.upload_file(prompt_host, prompt_remote)
            if invocation_id is not None:
                request_host = host_dir / "invocation.json"
                request_host.write_text(
                    json.dumps(
                        {
                            "invocation_id": invocation_id,
                            "role": role.value,
                            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                            "worktree": worktree,
                            "started_at": started.isoformat(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                await self.environment.upload_file(
                    request_host, f"{remote_dir}/invocation.json"
                )
                launcher_host = host_dir / "launcher.json"
                launcher_host.write_text(
                    json.dumps(
                        {
                            "schema_version": "v5-invocation-launcher-1",
                            "invocation_id": invocation_id,
                            "state": "PREPARED",
                            "pid": None,
                            "started_at": started.isoformat(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                await self.environment.upload_file(
                    launcher_host, f"{remote_dir}/launcher.json"
                )
        session_key = self._session_key(session_id, session_dir)
        if session_dir:
            await self.environment.exec(f"mkdir -p {shlex.quote(session_dir)}")
        if session_file is None and session_key is not None:
            session_file = self._session_paths.get(session_key)
            if session_file is None and session_dir is not None:
                session_file = await self._find_session_file(
                    session_dir, session_id or ""
                )
                if session_file is not None:
                    self._session_paths[session_key] = session_file
        pre_session_sha256 = await self._remote_sha256(session_file)
        command = _build_pi_command(
            model=self.model,
            prompt_path=prompt_remote,
            tools=tools,
            session_id=None if session_file else session_id,
            session_dir=None if session_file else session_dir,
            session_file=session_file,
            no_compaction=no_compaction,
        )
        wrapped = (
            "set -o pipefail; "
            f"{command} > {shlex.quote(transcript_remote)} "
            f"2> {shlex.quote(stderr_remote)}"
        )
        if invocation_id is not None:
            quoted_completion = shlex.quote(f"{remote_dir}/completion.json")
            quoted_completion_tmp = shlex.quote(f"{remote_dir}/completion.json.tmp")
            completion_format = (
                '{"invocation_id":"%s","exit_code":%s,'
                '"transcript_path":"%s","stderr_path":"%s",'
                '"transcript_sha256":"%s","stderr_sha256":"%s",'
                '"finished_at":"%s"}\\n'
            )
            wrapped = (
                "set +e; set -o pipefail; "
                # Persist process identity before Pi starts.  The record is
                # deliberately advisory: only completion.json plus file
                # hashes can authorize replay.
                f"started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
                f"printf '%s\\n' "
                f"'{{\"schema_version\":\"v5-invocation-launcher-1\",\"invocation_id\":\"{invocation_id}\",\"state\":\"RUNNING\",\"pid\":'\"$$\"',\"started_at\":\"'\"$started_at\"'\"}}' "
                f"> {shlex.quote(remote_dir + '/launcher.json.tmp')}; "
                f"mv -f -- {shlex.quote(remote_dir + '/launcher.json.tmp')} "
                f"{shlex.quote(remote_dir + '/launcher.json')}; "
                f"touch {shlex.quote(remote_dir + '/heartbeat')}; "
                f"{command} > {shlex.quote(transcript_remote)} "
                f"2> {shlex.quote(stderr_remote)}; "
                "invocation_status=$?; "
                f"transcript_sha256=$(sha256sum {shlex.quote(transcript_remote)} "
                "| awk '{print $1}'); "
                f"stderr_sha256=$(sha256sum {shlex.quote(stderr_remote)} "
                "| awk '{print $1}'); "
                "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
                f"printf {shlex.quote(completion_format)} "
                f"{shlex.quote(invocation_id)} \"$invocation_status\" "
                f"{shlex.quote(transcript_remote)} {shlex.quote(stderr_remote)} "
                '"$transcript_sha256" "$stderr_sha256" "$finished_at" '
                f"> {quoted_completion_tmp}; "
                f"mv -f -- {quoted_completion_tmp} {quoted_completion}; "
                f"printf '%s\\n' '{{\"state\":\"COMPLETED\",\"exit_code\":'\"$invocation_status\"'}}' > "
                f"{shlex.quote(remote_dir + '/state.json.tmp')}; "
                f"mv -f -- {shlex.quote(remote_dir + '/state.json.tmp')} "
                f"{shlex.quote(remote_dir + '/state.json')}; "
                'exit "$invocation_status"'
            )
        result: Any = None
        exception: Exception | None = None
        timed_out = False
        completion_record: dict[str, Any] | None = None
        indeterminate_error: IndeterminateInvocationError | None = None
        try:
            role_env = dict(self.env)
            if no_compaction:
                # Pi exposes auto-compaction as a settings key, not a CLI
                # switch.  Use one stable directory for all role invocations
                # in this runner so the setting is applied without touching a
                # reviewer worktree.
                settings_dir = "/external_eval/v5_pi_settings"
                settings = json.dumps(
                    {
                        "defaultProjectTrust": "never",
                        "telemetry": False,
                        "compaction": {"enabled": False},
                    },
                    separators=(",", ":"),
                )
                await self.environment.exec(
                    f"mkdir -p {shlex.quote(settings_dir)} && "
                    f"printf '%s' {shlex.quote(settings)} > "
                    f"{shlex.quote(settings_dir + '/settings.json')}"
                )
                role_env["PI_CODING_AGENT_DIR"] = settings_dir
            if invocation_replayed:
                completion_record = await self._read_completion_record(
                    remote_dir=remote_dir,
                    invocation_id=invocation_id or "",
                    transcript_path=transcript_remote,
                    stderr_path=stderr_remote,
                )

                class _ReplayedResult:
                    return_code = completion_record["exit_code"]

                result = _ReplayedResult()
            else:
                result = await self.environment.exec(
                    wrapped,
                    cwd=worktree,
                    env=role_env,
                    timeout_sec=int(timeout_sec) if timeout_sec is not None else None,
                )
        except (asyncio.TimeoutError, TimeoutError) as error:
            timed_out = True
            exception = error
        except Exception as error:  # noqa: BLE001 - preserve partial transcript
            exception = error

        if invocation_id is not None and completion_record is None:
            try:
                completion_record = await self._read_completion_record(
                    remote_dir=remote_dir,
                    invocation_id=invocation_id,
                    transcript_path=transcript_remote,
                    stderr_path=stderr_remote,
                )
                if (
                    result is not None
                    and getattr(result, "return_code", None)
                    != completion_record["exit_code"]
                ):
                    raise IndeterminateInvocationError(
                        f"Invocation {invocation_id} completion exit code conflicts "
                        "with the transport result"
                    )
                if result is None or exception is not None:

                    class _RecoveredResult:
                        return_code = completion_record["exit_code"]

                    result = _RecoveredResult()
                    exception = None
                    timed_out = False
            except Exception as completion_error:  # noqa: BLE001 - normalize trust failure
                indeterminate_error = (
                    completion_error
                    if isinstance(completion_error, IndeterminateInvocationError)
                    else IndeterminateInvocationError(
                        f"Invocation {invocation_id} has no trusted completion record: "
                        f"{completion_error}"
                    )
                )

        transcript_path = host_dir / "pi.jsonl"
        stderr_path = host_dir / "stderr.txt"
        session_host_path = host_dir / "session.jsonl"
        resolved_session_file = session_file
        transcript = await self._download_text(transcript_remote, transcript_path)
        stderr = await self._download_text(stderr_remote, stderr_path)
        if resolved_session_file is None and session_key is not None and session_dir is not None:
            resolved_session_file = await self._find_session_file(session_dir, session_id or "")
            if resolved_session_file is not None:
                self._session_paths[session_key] = resolved_session_file
        session_text = ""
        if resolved_session_file:
            session_text = await self._download_text(resolved_session_file, session_host_path)
        transcript = self.redact(transcript)
        stderr = self.redact(stderr)
        session_text = self.redact(session_text)
        transcript_path.write_text(transcript, encoding="utf-8")
        if resolved_session_file:
            session_host_path.write_text(session_text, encoding="utf-8")
        if stderr:
            stderr_path.write_text(stderr, encoding="utf-8")
        elif stderr_path.exists():
            stderr_path.unlink()
        if invocation_id is None:
            await self.environment.exec(f"rm -rf -- {shlex.quote(remote_dir)}")

        try:
            final_text = extract_pi_final_assistant_text(transcript)
        except ValueError:
            final_text = ""
        trajectory = pi_jsonl_to_atif(
            transcript,
            instruction=self.redact(prompt),
            model=self.model,
            agent_version=self.agent_version,
        )
        trajectory_path = host_dir / "trajectory.json"
        trajectory_path.write_text(
            format_trajectory_json(trajectory) + "\n",
            encoding="utf-8",
        )
        metrics = trajectory.get("final_metrics") or {}
        finished = datetime.now(timezone.utc)
        metrics = {
            **metrics,
            "wall_time_sec": (finished - started).total_seconds(),
            "test_command_count": self._test_command_count(trajectory),
            "invocation_replayed": invocation_replayed,
        }
        exit_code = 124 if timed_out else getattr(result, "return_code", 1)
        if exit_code == 0 and session_key is not None and resolved_session_file is None:
            exit_code = 1
            exception = RuntimeError("Pi did not create a discoverable persistent session file")
        detail = str(exception) if exception else (stderr[-2000:] if exit_code else None)
        tracked_diff_digest = await self._tracked_diff_digest(worktree)
        if indeterminate_error is not None:
            raise indeterminate_error
        return RoleRunResult(
            exit_code=exit_code,
            final_assistant_text=final_text,
            transcript_path=transcript_path,
            trajectory_path=trajectory_path,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            resource_usage=metrics,
            timed_out=timed_out,
            exception=self.redact(detail) if detail else None,
            tracked_diff_digest=tracked_diff_digest,
            session_file=session_host_path if resolved_session_file else None,
            pre_session_sha256=pre_session_sha256,
            post_session_sha256=await self._remote_sha256(resolved_session_file),
            invocation_id=invocation_id,
            remote_session_file=resolved_session_file,
        )

    @staticmethod
    def _session_key(session_id: str | None, session_dir: str | None) -> str | None:
        if not session_id or not session_dir:
            return None
        return f"{session_dir}\0{session_id}"

    async def _find_session_file(self, session_dir: str, session_id: str) -> str | None:
        """Resolve Pi's timestamped session filename after the first run."""
        pattern = f"*_{session_id}.jsonl"
        result = await self.environment.exec(
            f"find {shlex.quote(session_dir)} -maxdepth 1 -type f "
            f"-name {shlex.quote(pattern)} -print | sort"
        )
        if result.return_code != 0:
            return None
        candidates = (result.stdout or "").strip().splitlines()
        if len(candidates) > 1:
            raise IndeterminateInvocationError(
                f"Persistent Pi session ID {session_id} resolves to multiple files"
            )
        return candidates[0] if candidates else None

    async def _wait_for_invocation_completion(
        self, *, remote_dir: str, invocation_id: str
    ) -> None:
        """Wait briefly for a remote launcher already owning an invocation.

        This is intentionally bounded.  A missing completion after the wait
        remains an indeterminate invocation and is surfaced to the controller,
        which can resume the same identity later or explicitly expire it.
        """
        deadline = asyncio.get_running_loop().time() + self.invocation_recovery_wait_sec
        while asyncio.get_running_loop().time() < deadline:
            result = await self.environment.exec(
                f"test -f {shlex.quote(remote_dir + '/completion.json')}"
            )
            if result.return_code == 0:
                return
            await asyncio.sleep(min(0.25, max(0.01, deadline - asyncio.get_running_loop().time())))

    async def _read_invocation_state(self, remote_dir: str) -> str:
        """Return a diagnostic state without treating it as completion proof."""
        for name in ("state.json", "launcher.json"):
            try:
                result = await self.environment.exec(
                    f"cat {shlex.quote(remote_dir + '/' + name)}"
                )
            except Exception:  # noqa: BLE001 - state is diagnostic only
                continue
            if result.return_code != 0:
                continue
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                return "INVALID"
            state = payload.get("state")
            if isinstance(state, str) and state:
                return state
        return "UNKNOWN"

    async def _read_completion_record(
        self,
        *,
        remote_dir: str,
        invocation_id: str,
        transcript_path: str,
        stderr_path: str,
    ) -> dict[str, Any]:
        completion = await self.environment.exec(
            f"cat {shlex.quote(remote_dir + '/completion.json')}"
        )
        if completion.return_code != 0:
            raise IndeterminateInvocationError(
                f"Invocation {invocation_id} has no completion record"
            )
        try:
            record = json.loads(completion.stdout or "{}")
        except json.JSONDecodeError as error:
            raise IndeterminateInvocationError(
                f"Invocation {invocation_id} has an invalid completion record"
            ) from error
        expected = {
            "invocation_id": invocation_id,
            "transcript_path": transcript_path,
            "stderr_path": stderr_path,
        }
        if any(record.get(key) != value for key, value in expected.items()) or not isinstance(
            record.get("exit_code"), int
        ):
            raise IndeterminateInvocationError(
                f"Invocation {invocation_id} has an invalid completion record"
            )
        for field in ("transcript_sha256", "stderr_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field, ""))):
                raise IndeterminateInvocationError(
                    f"Invocation {invocation_id} has an invalid completion record"
                )
        try:
            finished_at = datetime.fromisoformat(str(record.get("finished_at", "")))
        except ValueError as error:
            raise IndeterminateInvocationError(
                f"Invocation {invocation_id} has an invalid completion timestamp"
            ) from error
        if finished_at.tzinfo is None:
            raise IndeterminateInvocationError(
                f"Invocation {invocation_id} completion timestamp has no timezone"
            )
        for path, digest_field in (
            (transcript_path, "transcript_sha256"),
            (stderr_path, "stderr_sha256"),
        ):
            if await self._remote_sha256(path) != record[digest_field]:
                raise IndeterminateInvocationError(
                    f"Invocation {invocation_id} completion record does not match its files"
                )
        return record

    async def _remote_sha256(self, remote_path: str | None) -> str | None:
        if not remote_path:
            return None
        result = await self.environment.exec(
            f"if test -f {shlex.quote(remote_path)}; then sha256sum {shlex.quote(remote_path)}; fi"
        )
        if result.return_code != 0 or not result.stdout:
            return None
        return result.stdout.strip().split()[0]

    async def _download_text(self, remote_path: str, host_path: Path) -> str:
        exists = await self.environment.exec(f"test -f {shlex.quote(remote_path)}")
        if exists.return_code != 0:
            return ""
        try:
            await self.environment.download_file(remote_path, host_path)
            return host_path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best-effort partial transcript retrieval
            return ""

    async def _tracked_diff_digest(self, worktree: str) -> str | None:
        repository = await self.environment.exec(
            "git rev-parse --is-inside-work-tree", cwd=worktree
        )
        if repository.return_code != 0:
            return None
        diff = await self.environment.exec(_temporary_index_diff_command("HEAD"), cwd=worktree)
        if diff.return_code != 0:
            return None
        return sha256_bytes((diff.stdout or "").encode("utf-8"))

    @staticmethod
    def _test_command_count(trajectory: dict[str, Any]) -> int:
        pattern = re.compile(
            r"(?:^|[;&|]\s*)(?:[A-Za-z0-9_./-]+\s+)*(?:pytest|go\s+test|cargo\s+test|"
            r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|mvn\s+test|"
            r"gradle\s+test|dotnet\s+test)\b",
            re.IGNORECASE,
        )
        count = 0
        for step in trajectory.get("steps", []):
            for call in step.get("tool_calls", []):
                if call.get("function_name") != "bash":
                    continue
                arguments = call.get("arguments") or {}
                command = arguments.get("command", arguments.get("cmd", ""))
                if isinstance(command, str) and pattern.search(command):
                    count += 1
        return count


class ContainerGitWorkspaceManager:
    """Create disposable role worktrees inside one Pier task environment."""

    ROOT = "/tmp/external_eval/worktrees"
    INPUT_ROOT = "/tmp/external_eval/contracts"

    def __init__(
        self,
        *,
        environment: BaseEnvironment,
        transfer_dir: Path,
        repository: str = "/app",
        redact=lambda value: value,
    ) -> None:
        self.environment = environment
        self.transfer_dir = transfer_dir
        self.repository = repository
        self.redact = redact
        self._prepared = False
        self._managed_worktrees: list[str] = []

    async def prepare(self, *, base_commit: str) -> WorkflowWorktrees:
        head = await self._checked("git rev-parse HEAD", cwd=self.repository)
        if (head.stdout or "").strip() != base_commit:
            raise ValueError(
                f"Repository HEAD {(head.stdout or '').strip()} does not match base {base_commit}"
            )
        await self._checked(f"mkdir -p {shlex.quote(self.ROOT)} {shlex.quote(self.INPUT_ROOT)}")
        await self._checked("git worktree prune", cwd=self.repository)
        paths = WorkflowWorktrees(
            solver=f"{self.ROOT}/solver",
            spec_review=f"{self.ROOT}/spec-review",
            patch_review=f"{self.ROOT}/patch-review",
            repair=f"{self.ROOT}/repair",
            correction=f"{self.ROOT}/correction",
            final=self.repository,
            summary=f"{self.ROOT}/summary",
        )
        self._prepared = True
        try:
            for path in (
                paths.solver,
                paths.spec_review,
                paths.patch_review,
                paths.repair,
                paths.summary,
            ):
                assert path is not None
                exists = await self.environment.exec(f"test -e {shlex.quote(path)}")
                if exists.return_code == 0:
                    raise RuntimeError(f"Refusing to replace pre-existing worktree path {path}")
                await self._checked(
                    f"git worktree add --detach {shlex.quote(path)} {shlex.quote(base_commit)}",
                    cwd=self.repository,
                )
                self._managed_worktrees.append(path)
            # Git worktrees provide independent working-tree files.  Refuse a
            # task image that was assembled with ``cp -al``/hard links: a
            # Reviewer write in such a tree would mutate /app or another
            # Reviewer despite the separate path names.
            for path in (
                paths.spec_review,
                paths.patch_review,
                paths.summary,
            ):
                if path is not None:
                    await self._assert_no_hardlinks(path)
            await self._checked(f"mkdir -p {shlex.quote(paths.correction)}")
        except Exception:
            await self.cleanup()
            raise
        return paths

    async def _assert_no_hardlinks(self, worktree: str) -> None:
        result = await self._checked(
            f"find {shlex.quote(worktree)} -type f -links +1 "
            f"-not -path {shlex.quote(worktree + '/.git/*')} -print"
        )
        linked = [line for line in (result.stdout or "").splitlines() if line]
        if linked:
            raise RuntimeError(
                "Reviewer worktree is not copy-on-write isolated; hard-linked files: "
                + ", ".join(linked[:20])
            )

    async def prepare_additional_worktree(self, *, name: str, base_commit: str) -> str:
        if not self._prepared:
            raise RuntimeError("Primary workflow worktrees must be prepared first")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Unsafe worktree name: {name}")
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", base_commit):
            raise ValueError("Invalid worktree base commit")
        path = f"{self.ROOT}/{name}"
        exists = await self.environment.exec(f"test -e {shlex.quote(path)}")
        if exists.return_code == 0:
            raise RuntimeError(f"Refusing to replace pre-existing worktree path {path}")
        await self._checked(
            f"git worktree add --detach {shlex.quote(path)} {shlex.quote(base_commit)}",
            cwd=self.repository,
        )
        self._managed_worktrees.append(path)
        return path

    async def apply_patch(self, *, worktree: str, patch: bytes) -> None:
        if not patch:
            return
        patch_path = await self.materialize_file(
            name=f"apply-{sha256_bytes(patch)[:16]}.patch", payload=patch
        )
        await self._checked(
            f"git apply --check --whitespace=nowarn {shlex.quote(patch_path)}",
            cwd=worktree,
        )
        await self._checked(
            f"git apply --whitespace=nowarn {shlex.quote(patch_path)}", cwd=worktree
        )

    async def discard_worktree(self, *, worktree: str) -> None:
        if not worktree.startswith(f"{self.ROOT}/"):
            raise ValueError(f"Refusing to discard worktree outside {self.ROOT}: {worktree}")
        await self._checked(
            f"git worktree remove --force {shlex.quote(worktree)}", cwd=self.repository
        )

    async def materialize_file(self, *, name: str, payload: bytes, readonly: bool = False) -> str:
        if Path(name).name != name:
            raise ValueError("Contract materialization name must be a plain file name")
        self.transfer_dir.mkdir(parents=True, exist_ok=True)
        host = self.transfer_dir / name
        host.write_bytes(payload)
        remote = f"{self.INPUT_ROOT}/{name}"
        await self.environment.upload_file(host, remote)
        if readonly:
            await self._checked(f"chmod 0444 {shlex.quote(remote)}")
        return remote

    async def materialized_file_digest(self, *, path: str) -> str:
        if not path.startswith(f"{self.INPUT_ROOT}/"):
            raise ValueError(f"Refusing to hash a file outside {self.INPUT_ROOT}: {path}")
        result = await self._checked(f"sha256sum -- {shlex.quote(path)}")
        digest = (result.stdout or "").split(maxsplit=1)[0]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Could not read SHA-256 for frozen input {path}")
        return digest

    async def read_materialized_file(self, *, path: str) -> bytes:
        if not path.startswith(f"{self.INPUT_ROOT}/"):
            raise ValueError(f"Refusing to read a file outside {self.INPUT_ROOT}: {path}")
        self.transfer_dir.mkdir(parents=True, exist_ok=True)
        host_path = self.transfer_dir / f"read-{Path(path).name}"
        try:
            await self.environment.download_file(path, host_path)
            return host_path.read_bytes()
        finally:
            host_path.unlink(missing_ok=True)

    async def capture_patch(
        self,
        *,
        worktree: str,
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> bytes:
        if baseline_patch:
            baseline_path = await self.materialize_file(
                name=f"baseline-{sha256_bytes(baseline_patch)[:16]}.patch",
                payload=baseline_patch,
            )
            quoted_base = shlex.quote(base_commit)
            quoted_baseline = shlex.quote(baseline_path)
            command = (
                'snapshot_dir="$(mktemp -d)" && '
                "trap 'rm -rf -- \"$snapshot_dir\"' EXIT && "
                'baseline_index="$snapshot_dir/baseline-index" && '
                'current_index="$snapshot_dir/current-index" && '
                f'GIT_INDEX_FILE="$baseline_index" git read-tree {quoted_base} && '
                f'GIT_INDEX_FILE="$baseline_index" git apply --cached --whitespace=nowarn {quoted_baseline} && '
                'baseline_tree="$(GIT_INDEX_FILE="$baseline_index" git write-tree)" && '
                f'GIT_INDEX_FILE="$current_index" git read-tree {quoted_base} && '
                'GIT_INDEX_FILE="$current_index" git add -A -- . && '
                'GIT_INDEX_FILE="$current_index" git diff --cached --binary '
                '--no-ext-diff "$baseline_tree" -- .'
            )
            result = await self._checked(command, cwd=worktree)
            return (result.stdout or "").encode("utf-8")
        result = await self._checked(_temporary_index_diff_command(base_commit), cwd=worktree)
        return (result.stdout or "").encode("utf-8")

    async def baseline_manifest(
        self, *, worktree: str, base_commit: str, patch: bytes | None = None
    ) -> dict[str, Any]:
        """Return the canonical environment baseline used by publication.

        A patch digest alone hides which files made the task image dirty.  The
        manifest binds the exact base object, complete patch, and path list so
        a later publish can explain a mismatch and distinguish environment
        setup edits from Reviewer pollution.
        """
        resolved = await self._checked(
            f"git rev-parse --verify {shlex.quote(base_commit + '^{commit}')}",
            cwd=worktree,
        )
        resolved_commit = (resolved.stdout or "").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved_commit):
            raise RuntimeError("Could not resolve the baseline base commit")
        if patch is None:
            patch = await self.capture_patch(worktree=worktree, base_commit=base_commit)
        status = await self._checked(
            "git status --porcelain=v1 --untracked-files=all", cwd=worktree
        )
        paths: list[str] = []
        for line in (status.stdout or "").splitlines():
            if not line:
                continue
            value = line[3:] if len(line) >= 4 else line
            # Rename entries use ``old -> new``; bind both paths so a
            # publication error identifies the complete dirty footprint.
            paths.extend(part.strip() for part in value.split(" -> ") if part.strip())
        return {
            "schema_version": "v5-baseline-1",
            "base_commit": resolved_commit,
            "patch_sha256": sha256_bytes(patch),
            "paths": sorted(set(paths)),
            "status_porcelain_sha256": sha256_bytes(
                (status.stdout or "").encode("utf-8")
            ),
        }

    async def tracked_diff_digest(
        self,
        *,
        worktree: str,
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> str:
        return sha256_bytes(
            await self.capture_patch(
                worktree=worktree,
                base_commit=base_commit,
                baseline_patch=baseline_patch,
            )
        )

    async def tracked_files_diff_digest(self, *, worktree: str, base_commit: str) -> str:
        result = await self._checked(
            _temporary_index_tracked_diff_command(base_commit), cwd=worktree
        )
        return sha256_bytes((result.stdout or "").encode("utf-8"))

    async def restore_reviewer_worktree(
        self,
        *,
        worktree: str,
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> bytes:
        """Discard reviewer scratch files and restore its disposable worktree."""
        if not worktree.startswith(f"{self.ROOT}/"):
            raise ValueError(
                f"Refusing to restore reviewer worktree outside {self.ROOT}: {worktree}"
            )
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", base_commit):
            raise ValueError("Refusing to restore reviewer worktree with an invalid base commit")
        discarded = await self.capture_patch(
            worktree=worktree,
            base_commit=base_commit,
            baseline_patch=baseline_patch,
        )
        await self._checked(
            f"git reset --hard {shlex.quote(base_commit)}",
            cwd=worktree,
        )
        await self._checked("git clean -fdx -- .", cwd=worktree)
        if baseline_patch:
            await self.apply_patch(worktree=worktree, patch=baseline_patch)
        return discarded

    async def validate_evidence(self, *, worktree: str, assessment: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        evidence_items: list[tuple[str, int | None, str]] = []
        assessment_items = assessment.get("obligation_assessments") or assessment.get(
            "criterion_assessments", []
        )
        assessment_items = [
            *assessment_items,
            *(assessment.get("late_obligation_candidates") or []),
        ]
        for item in assessment_items:
            ref = (
                item.get("obligation_id")
                or item.get("criterion_ref")
                or item.get("obligation_ref")
                or "assessment"
            )
            for evidence in item["implementation_evidence"]:
                evidence_items.append((evidence["path"], evidence["line_end"], ref))
            for evidence in item.get("test_evidence", []):
                evidence_items.append((evidence["path"], None, ref))
        for regression in assessment.get("unexpected_regressions") or []:
            ref = regression.get("id") or "unexpected_regression"
            for evidence in regression["implementation_evidence"]:
                evidence_items.append((evidence["path"], evidence["line_end"], ref))
        cache: dict[str, int | None] = {}
        for path, line_end, ref in evidence_items:
            if path not in cache:
                result = await self.environment.exec(
                    f"test -f {shlex.quote(path)} && test ! -L {shlex.quote(path)} "
                    f"&& awk 'END {{ print NR }}' {shlex.quote(path)}",
                    cwd=worktree,
                )
                if result.return_code == 0:
                    try:
                        cache[path] = int((result.stdout or "").strip())
                    except ValueError:
                        cache[path] = None
                else:
                    cache[path] = None
            lines = cache[path]
            if lines is None:
                errors.append(f"{ref} evidence path does not exist: {path}")
            elif line_end is not None and line_end > lines:
                errors.append(f"{ref} evidence line {line_end} exceeds {path} length {lines}")
        return errors

    async def validate_v5_code_references(
        self,
        *,
        base_worktree: str,
        candidate_worktree: str,
        base_commit: str,
        assessments: list[dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        for assessment in assessments:
            target_id = assessment.get("target_id", "assessment")
            for reference in assessment.get("code_refs", []):
                raw_path = reference.get("path", "")
                path = Path(raw_path)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    errors.append(f"{target_id} code reference has unsafe path: {raw_path}")
                    continue
                line_end = reference["line_end"]
                if reference["snapshot"] == "base":
                    tree = await self.environment.exec(
                        f"git ls-tree {shlex.quote(base_commit)} -- {shlex.quote(raw_path)}",
                        cwd=base_worktree,
                    )
                    tree_line = (tree.stdout or "").strip()
                    if tree.return_code != 0 or not tree_line:
                        errors.append(f"{target_id} base code reference is missing: {raw_path}")
                        continue
                    mode = tree_line.split(maxsplit=1)[0]
                    if mode == "120000" or not mode.startswith("100"):
                        errors.append(
                            f"{target_id} base code reference is not a regular file: {raw_path}"
                        )
                        continue
                    content = await self.environment.exec(
                        f"git show {shlex.quote(base_commit + ':' + raw_path)}",
                        cwd=base_worktree,
                    )
                    line_count = len((content.stdout or "").splitlines())
                    if content.return_code != 0 or line_end > line_count:
                        errors.append(
                            f"{target_id} base code reference range is invalid: {raw_path}"
                        )
                else:
                    result = await self.environment.exec(
                        "root_real=$(realpath -- .) && "
                        f"candidate_real=$(realpath -- {shlex.quote(raw_path)}) && "
                        'case "$candidate_real" in "$root_real"/*) ;; '
                        "*) exit 1 ;; esac; "
                        f"test -f {shlex.quote(raw_path)} && "
                        f"test ! -L {shlex.quote(raw_path)} && "
                        f"awk 'END {{ print NR }}' {shlex.quote(raw_path)}",
                        cwd=candidate_worktree,
                    )
                    try:
                        line_count = int((result.stdout or "").strip())
                    except ValueError:
                        line_count = 0
                    if result.return_code != 0 or line_end > line_count:
                        errors.append(
                            f"{target_id} candidate code reference range is invalid: {raw_path}"
                        )
        return errors

    async def validate_repository_sources(
        self,
        *,
        worktree: str,
        base_commit: str,
        sources: list[dict[str, Any]],
    ) -> list[str]:
        """Validate repository source anchors against the frozen base tree."""
        errors: list[str] = []
        frozen_revision = await self.environment.exec(
            f"git rev-parse --verify {shlex.quote(base_commit + '^{commit}')}",
            cwd=worktree,
        )
        frozen_object = (frozen_revision.stdout or "").strip()
        if frozen_revision.return_code != 0 or not re.fullmatch(
            r"[0-9a-fA-F]{40,64}", frozen_object
        ):
            return ["The frozen task base commit cannot be resolved"]
        revision_cache: dict[str, str | None] = {}
        for source in sources:
            if source.get("kind") != "repository":
                continue
            source_id = source.get("id", "source")
            locator = source.get("locator") or {}
            path = locator.get("path", "")
            revision = locator.get("base_revision", "")
            if revision not in revision_cache:
                if not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
                    revision_cache[revision] = None
                else:
                    resolved = await self.environment.exec(
                        f"git rev-parse --verify {shlex.quote(revision + '^{commit}')}",
                        cwd=worktree,
                    )
                    candidate = (resolved.stdout or "").strip()
                    revision_cache[revision] = (
                        candidate
                        if resolved.return_code == 0
                        and re.fullmatch(r"[0-9a-fA-F]{40,64}", candidate)
                        else None
                    )
            if revision_cache[revision] != frozen_object:
                errors.append(f"{source_id} base revision does not match the frozen task base")
                continue
            tree = await self.environment.exec(
                f"git ls-tree {shlex.quote(base_commit)} -- {shlex.quote(path)}",
                cwd=worktree,
            )
            tree_line = (tree.stdout or "").strip()
            if tree.return_code != 0 or not tree_line:
                errors.append(f"{source_id} repository source does not exist at base: {path}")
                continue
            mode = tree_line.split(maxsplit=1)[0]
            if mode == "120000" or not mode.startswith("100"):
                errors.append(f"{source_id} repository source is not a regular file: {path}")
                continue
            content_result = await self.environment.exec(
                f"git show {shlex.quote(base_commit + ':' + path)}",
                cwd=worktree,
            )
            if content_result.return_code != 0:
                errors.append(f"{source_id} repository source cannot be read: {path}")
                continue
            content = (content_result.stdout or "").encode("utf-8")
            if sha256_bytes(content) != locator.get("file_sha256"):
                errors.append(f"{source_id} repository source digest does not match: {path}")
            lines = content.decode("utf-8", errors="replace").splitlines()
            start = locator.get("line_start")
            end = locator.get("line_end")
            if not isinstance(start, int) or not isinstance(end, int) or end > len(lines):
                errors.append(f"{source_id} repository source line range is invalid: {path}")
                continue
            actual_quote = "\n".join(lines[start - 1 : end])
            if actual_quote != locator.get("quote"):
                errors.append(f"{source_id} repository source quote does not match: {path}")
        return errors

    async def bind_repository_sources(
        self,
        *,
        worktree: str,
        base_commit: str,
        sources: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Validate V3 source anchors and add deterministic frozen-base bindings."""
        errors: list[str] = []
        bound: list[dict[str, Any]] = []
        frozen_revision = await self.environment.exec(
            f"git rev-parse --verify {shlex.quote(base_commit + '^{commit}')}",
            cwd=worktree,
        )
        frozen_object = (frozen_revision.stdout or "").strip()
        if frozen_revision.return_code != 0 or not re.fullmatch(
            r"[0-9a-fA-F]{40,64}", frozen_object
        ):
            return [], ["The frozen task base commit cannot be resolved"]
        for source in sources:
            if source.get("kind") == "task":
                bound.append(dict(source))
                continue
            source_id = source.get("id", "source")
            path = source.get("path", "")
            tree = await self.environment.exec(
                f"git ls-tree {shlex.quote(base_commit)} -- {shlex.quote(path)}",
                cwd=worktree,
            )
            tree_line = (tree.stdout or "").strip()
            if tree.return_code != 0 or not tree_line:
                errors.append(f"{source_id} repository source does not exist at base: {path}")
                continue
            mode = tree_line.split(maxsplit=1)[0]
            if mode == "120000" or not mode.startswith("100"):
                errors.append(f"{source_id} repository source is not a regular file: {path}")
                continue
            content_result = await self.environment.exec(
                f"git show {shlex.quote(base_commit + ':' + path)}",
                cwd=worktree,
            )
            if content_result.return_code != 0:
                errors.append(f"{source_id} repository source cannot be read: {path}")
                continue
            content = (content_result.stdout or "").encode("utf-8")
            digest_result = await self.environment.exec(
                f"git show {shlex.quote(base_commit + ':' + path)} | sha256sum",
                cwd=worktree,
            )
            file_sha256 = (digest_result.stdout or "").split(maxsplit=1)[0]
            if digest_result.return_code != 0 or not re.fullmatch(
                r"[0-9a-f]{64}", file_sha256
            ):
                errors.append(f"{source_id} repository source digest cannot be read: {path}")
                continue
            lines = content.decode("utf-8", errors="replace").splitlines()
            start = source.get("line_start")
            end = source.get("line_end")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
                or end > len(lines)
            ):
                errors.append(f"{source_id} repository source line range is invalid: {path}")
                continue
            bound.append(
                {
                    "id": source_id,
                    "kind": "repository",
                    "path": path,
                    "line_start": start,
                    "line_end": end,
                    "quote": "\n".join(lines[start - 1 : end]),
                    "base_revision": frozen_object,
                    "file_sha256": file_sha256,
                }
            )
        return bound, errors

    async def publish_final_patch(
        self,
        *,
        base_commit: str,
        patch: bytes,
        baseline_patch: bytes = b"",
        baseline_manifest: dict[str, Any] | None = None,
    ) -> None:
        """Apply P1 only when /app still matches its recorded baseline diff."""
        head = await self._checked("git rev-parse HEAD", cwd=self.repository)
        if (head.stdout or "").strip() != base_commit:
            raise RuntimeError("Cannot publish P1: /app no longer points to the base commit")
        current_digest = await self.tracked_diff_digest(
            worktree=self.repository, base_commit=base_commit
        )
        baseline_digest = sha256_bytes(baseline_patch)
        if current_digest != baseline_digest:
            current_manifest = await self.baseline_manifest(
                worktree=self.repository, base_commit=base_commit
            )
            paths = ", ".join(current_manifest["paths"][:20]) or "unknown paths"
            raise RuntimeError(
                "Cannot publish P1 into a modified /app worktree outside the "
                f"recorded baseline (modified paths: {paths})"
            )
        if baseline_manifest is not None:
            expected_base = str(baseline_manifest.get("base_commit", ""))
            if expected_base and expected_base != base_commit:
                raise RuntimeError("Cannot publish P1: baseline manifest targets another commit")
            if baseline_manifest.get("patch_sha256") != baseline_digest:
                raise RuntimeError("Cannot publish P1: baseline patch does not match its manifest")
            current_manifest = await self.baseline_manifest(
                worktree=self.repository, base_commit=base_commit, patch=baseline_patch
            )
            if current_manifest["paths"] != sorted(set(baseline_manifest.get("paths", []))):
                raise RuntimeError(
                    "Cannot publish P1: /app baseline paths changed outside the recorded manifest"
                )
        # A task image may have already applied the exact P0/P1 diff during
        # image construction (the numba compatibility edit is a concrete
        # example).  Re-applying the identical patch would fail even though
        # the resulting tree is correct; treat it as an idempotent publish.
        if patch and current_digest == sha256_bytes(patch):
            return
        if patch:
            patch_path = await self.materialize_file(name="p1.patch", payload=patch)
            await self._checked(
                f"git apply --check --whitespace=nowarn {shlex.quote(patch_path)}",
                cwd=self.repository,
            )
            await self._checked(
                f"git apply --whitespace=nowarn {shlex.quote(patch_path)}",
                cwd=self.repository,
            )

    async def restore_final_worktree(self, *, base_commit: str) -> bytes:
        """Restore the ephemeral task repository and return the discarded diff."""
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", base_commit):
            raise ValueError("Refusing to restore /app with an invalid base commit")
        discarded = await self.capture_patch(
            worktree=self.repository,
            base_commit=base_commit,
        )
        await self._checked(
            f"git reset --hard {shlex.quote(base_commit)}",
            cwd=self.repository,
        )
        await self._checked("git clean -fd -- .", cwd=self.repository)
        return discarded

    async def cleanup(self) -> None:
        if not self._prepared:
            return
        for path in reversed(self._managed_worktrees):
            await self.environment.exec(
                f"git worktree remove --force {shlex.quote(path)}", cwd=self.repository
            )
        await self.environment.exec("git worktree prune", cwd=self.repository)
        self._managed_worktrees.clear()
        self._prepared = False

    async def _checked(self, command: str, *, cwd: str | None = None):
        result = await self.environment.exec(command, cwd=cwd)
        if result.return_code != 0:
            output = self.redact(result.stderr or result.stdout or "no output")
            raise RuntimeError(f"Command exited with status {result.return_code}: {output[-2000:]}")
        return result
