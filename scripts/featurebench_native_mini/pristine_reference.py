"""CPU capability probe against pristine source, without Level-1 masking.

Runs only in a private evaluation container. No source, tests or outputs from
this probe are copied into a Solver container or included in its prompt.
"""
from __future__ import annotations

import io
import logging
import shlex
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def summarize_junit(task, exit_code, reports):
    counts = {"tests_collected": 0, "tests_executed": 0, "tests_failed": 0,
              "tests_errors": 0, "tests_skipped": 0}
    error = None
    try:
        if not reports:
            raise ValueError("Pristine probe produced no JUnit report")
        for report in reports:
            cases = list(ET.fromstring(report).iter("testcase"))
            counts["tests_collected"] += len(cases)
            for case in cases:
                if case.find("skipped") is not None:
                    counts["tests_skipped"] += 1
                    continue
                counts["tests_executed"] += 1
                counts["tests_failed"] += case.find("failure") is not None
                counts["tests_errors"] += case.find("error") is not None
    except (ValueError, ET.ParseError) as exc:
        error = str(exc)
    passed = bool(exit_code == 0 and not error and counts["tests_executed"] > 0
                  and not counts["tests_failed"] and not counts["tests_errors"])
    return {"instance_id": task["instance_id"], "stage": "reference",
            "completed": True, "error": error, "probe_passed": passed,
            "cpu_reference_kind": "pristine_original_tests", "benchmark_score": False,
            "original_level": task["level"], "scored_level": None,
            "test_exit_code": exit_code, "cpu_override": True,
            "network": "none", "model_api_calls": 0, **counts}


def run_pristine_reference(task, output, runtime, manager_class):
    # Import the pinned official helpers only when running a real probe, so the
    # report validation can be tested independently of the vendor checkout.
    from featurebench.harness.container import exec_run_with_timeout
    from featurebench.harness.test_parsers import MAP_REPO_TO_TEST_CMD
    from featurebench.harness.utils import (
        build_test_command,
        get_test_config_from_repo_settings,
        parse_repo_settings,
    )
    from featurebench.utils.command_utils import apply_uv_run_prefix

    import docker

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("featurebench.pristine")
    manager = manager_class(logger)
    container = None
    prefix = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "
    try:
        container = manager.create_container(task["image_name"], task["instance_id"] + "-pristine",
                                             docker_runtime_config=runtime)
        code, data = exec_run_with_timeout(
            container, prefix + "rm -rf /testbed/* && cp -r /root/my_repo/* /testbed/", timeout=600)
        (output / "restore-output.txt").write_bytes(data)
        if code:
            raise RuntimeError("Could not restore the pristine project; see restore-output.txt")
        settings = get_test_config_from_repo_settings(parse_repo_settings(task))
        base = MAP_REPO_TO_TEST_CMD.get(task["repo"]) or settings["test_cmd"]
        command = apply_uv_run_prefix(build_test_command(base, settings["timeout_one"]), settings)
        paths = task["FAIL_TO_PASS"]
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError("Pristine probe has no target test files")
        files = [path if path.startswith("/testbed/") else "/testbed/" + path for path in paths]
        reports, codes, commands = [], [], []
        # Official P2P files run in separate pytest processes. Preserve that
        # isolation when this helper diagnoses a Level-1 reference failure.
        with (output / "test-output.txt").open("wb") as log:
            for number, path in enumerate(files):
                junit = f"/tmp/featurebench-pristine-{number}.xml"
                invocation = prefix + "cd /testbed && " + command + " " + shlex.quote(path)
                invocation += " --junitxml=" + junit
                commands.append(invocation)
                code, data = exec_run_with_timeout(container, invocation, timeout=settings.get("timeout_run", 1800))
                codes.append(code)
                log.write((f"\n===== {path} =====\n").encode() + data)
                log.flush()
                try:
                    chunks, _ = container.get_archive(junit)
                    with tarfile.open(fileobj=io.BytesIO(b"".join(chunks))) as archive:
                        report = archive.extractfile(next(m for m in archive.getmembers() if m.isfile())).read()
                    reports.append(report)
                    (output / f"junit-{number}.xml").write_bytes(report)
                except docker.errors.NotFound:
                    pass  # Missing reports explicitly fail the readiness check.
        (output / "test-command.txt").write_text("\n".join(commands) + "\n")
        summary = summarize_junit(task, next((code for code in codes if code != 0), 0), reports)
        summary["test_files"] = files
        if len(reports) != len(files):
            summary.update(probe_passed=False, error="Missing JUnit report for at least one target file")
        return summary
    finally:
        if container is not None:
            container.remove(force=True)
        manager.client.close()
