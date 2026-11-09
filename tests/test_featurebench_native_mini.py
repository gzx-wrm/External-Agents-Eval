from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/featurebench_native_mini/common.py"
if not SOURCE.exists():
    SOURCE = ROOT / "scripts/common.py"
spec = importlib.util.spec_from_file_location("featurebench_native_common", SOURCE)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
probe_spec = importlib.util.spec_from_file_location("featurebench_pristine_probe", SOURCE.parent / "pristine_reference.py")
pristine = importlib.util.module_from_spec(probe_spec)
probe_spec.loader.exec_module(pristine)


class CPUReferenceTests(unittest.TestCase):
    def test_level2_reference_does_not_change_real_task_or_claim_gold(self):
        task = {"level": 2, "patch": "", "test_patch": "modify tests to call agent_code",
                "PASS_TO_PASS": ["existing.py"], "FAIL_TO_PASS": ["feature.py"]}
        before = copy.deepcopy(task)
        reference = common.cpu_reference_task(task)
        self.assertEqual(reference, before)
        self.assertEqual(task, before)
        self.assertIsNot(reference, task)

    def test_empty_or_errored_reference_is_not_cpu_readiness(self):
        valid = {"completed": True, "resolved": True, "f2p_total": 10, "error": None}
        self.assertTrue(common.positive_reference_passed(valid))
        self.assertFalse(common.positive_reference_passed({**valid, "f2p_total": 0}))
        self.assertFalse(common.positive_reference_passed({**valid, "error": "Docker failed"}))
        self.assertFalse(common.positive_reference_passed({**valid, "resolved": False}))

    def test_gpu_dependency_and_unrelated_environment_faults_remain_distinct(self):
        self.assertEqual(common.classify_reference_failure("RuntimeError: Found no NVIDIA driver on your system"),
                         "gpu_deferred")
        self.assertEqual(common.classify_reference_failure("RuntimeError: 0 active drivers ([]). There should only be one."),
                         "gpu_deferred")
        self.assertEqual(common.classify_reference_failure("ModuleNotFoundError: missing_package"),
                         "cpu_validation_failed")
        self.assertEqual(common.classify_reference_failure("Importing CUDA-compatible torch; connection timed out"),
                         "cpu_validation_failed")

    def test_pristine_probe_counts_real_execution_without_claiming_a_score(self):
        xml = '<testsuite><testcase name="a"/><testcase name="b"/><testcase name="skip"><skipped/></testcase></testsuite>'
        summary = pristine.summarize_junit({"instance_id": "example.lv2", "level": 2}, 0, [xml])
        self.assertEqual(summary["tests_executed"], 2)
        self.assertEqual(summary["tests_skipped"], 1)
        self.assertIsNone(summary["scored_level"])
        self.assertFalse(summary["benchmark_score"])
        self.assertTrue(common.positive_reference_passed(summary))

    def test_pristine_probe_rejects_missing_empty_skipped_failed_and_timed_out_runs(self):
        task = {"instance_id": "example.lv2", "level": 2}
        samples = [
            (0, []), (0, ['<testsuite/>']),
            (0, ['<testsuite><testcase><skipped/></testcase></testsuite>']),
            (0, ['<testsuite><testcase><failure/></testcase></testsuite>']),
            (0, ['<testsuite><testcase><error/></testcase></testsuite>']),
            (124, ['<testsuite><testcase/></testsuite>']), (0, ['not xml']),
        ]
        for code, reports in samples:
            with self.subTest(code=code, reports=reports):
                self.assertFalse(common.positive_reference_passed(pristine.summarize_junit(task, code, reports)))

    def test_level2_reference_uses_probe_but_model_and_control_keep_official_level2(self):
        task = {"instance_id": "example.lv2", "level": 2, "patch": "",
                "test_patch": "forward test rewrite", "FAIL_TO_PASS": ["feature.py"], "PASS_TO_PASS": []}
        evaluations = []
        official = ModuleType("featurebench.harness.run_evaluation")

        def evaluate(instance, prediction, output, **kwargs):
            evaluations.append(copy.deepcopy(instance.to_dict() if isinstance(instance, pd.Series) else instance))
            return {"completed": True, "resolved": False, "patch_applied": True,
                    "report": {task["instance_id"]: {"tests_status": {
                        "FAIL_TO_PASS": {"success": [], "failure": ["feature.py"]}}}}}

        official.run_instance = evaluate
        official.EvalContainerManager = object
        container = ModuleType("featurebench.harness.container")
        container.EvalContainerManager = object
        utils = ModuleType("featurebench.harness.utils")
        utils.preprocess_hf_patch = lambda *args: self.fail("Lv2 must not use a gold corruption patch")
        models = ModuleType("featurebench.infer.models")
        models.TaskInstance = SimpleNamespace(from_dict=lambda value: SimpleNamespace(
            get_docker_runtime_config=lambda: {"need_gpu": True}))
        harness = ModuleType("featurebench.harness")
        harness.run_evaluation = official
        stubs = {"common": common, "pristine_reference": pristine,
                 "featurebench": ModuleType("featurebench"), "featurebench.harness": harness,
                 "featurebench.harness.run_evaluation": official,
                 "featurebench.harness.container": container, "featurebench.harness.utils": utils,
                 "featurebench.infer": ModuleType("featurebench.infer"), "featurebench.infer.models": models}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, stubs):
            root = Path(folder)
            (root / "task.json").write_text(json.dumps(task))
            (root / "model.patch").write_text("independent implementation")
            spec = importlib.util.spec_from_file_location("featurebench_evaluate_test", SOURCE.parent / "evaluate.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch.object(module, "run_pristine_reference", return_value={"probe_passed": True}) as probe:
                for stage in ["reference", "control", "model"]:
                    args = ["evaluate.py", "--task", str(root / "task.json"), "--output", str(root / stage),
                            "--stage", stage, "--model-patch", str(root / "model.patch")]
                    with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                probe.assert_called_once()
                self.assertEqual(probe.call_args.args[0], task)
        self.assertEqual(evaluations, [task, task])


class OfficialImageTransportTests(unittest.TestCase):
    def run_cached_image(self, directory, actual_id):
        repository = "libercoders/example"
        record = {"original_repository": repository, "original_digest": "sha256:manifest",
                  "config_digest": "sha256:original-config", "mirror_reference": "mirror/example@sha256:manifest"}
        metadata = directory / "preflight/images/example.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps(record))
        looked_up, tagged = [], []
        image = SimpleNamespace(id=actual_id, tag=lambda *args, **kwargs: tagged.append((args, kwargs)))

        def get(reference):
            looked_up.append(reference)
            return image

        client = SimpleNamespace(images=SimpleNamespace(get=get), close=lambda: None)
        source = SOURCE.parent / "run.py"
        code = source.read_text()
        node = next(item for item in ast.parse(code).body
                    if isinstance(item, ast.FunctionDef) and item.name == "ensure_image")
        namespace = {"Path": Path, "json": json, "save": common.save, "now": common.now,
                     "docker": SimpleNamespace(from_env=lambda **kwargs: client)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
        return namespace["ensure_image"], looked_up, tagged

    def test_official_route_keeps_original_manifest_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure, looked_up, tagged = self.run_cached_image(root, "sha256:original-config")
            self.assertEqual(ensure(root, "libercoders/example"), "sha256:original-config")
            self.assertEqual(looked_up, ["libercoders/example@sha256:manifest"])
            self.assertEqual(tagged, [(("libercoders/example",), {"tag": "latest"})])

    def test_image_with_different_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure, _, tagged = self.run_cached_image(root, "sha256:wrong-config")
            with self.assertRaisesRegex(ValueError, "configuration digest"):
                ensure(root, "libercoders/example")
            self.assertEqual(tagged, [])


if __name__ == "__main__":
    unittest.main()
