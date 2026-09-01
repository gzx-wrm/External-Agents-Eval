import json

from external_agents_eval.report import load_predictions, remove_test_changes


def test_remove_test_changes_keeps_implementation():
    patch = """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-old
+new
diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
@@ -1 +1 @@
-old test
+new test
"""
    filtered = remove_test_changes(patch)
    assert "pkg/a.py" in filtered
    assert "tests/test_a.py" not in filtered


def test_load_predictions_filters_to_fixed_manifest(tmp_path):
    for instance_id in ("selected", "extra"):
        (tmp_path / f"{instance_id}.json").write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": "agent",
                    "model_patch": "patch",
                    "generation_status": "success",
                }
            )
        )
    assert set(load_predictions(tmp_path, ("selected",))) == {"selected"}
