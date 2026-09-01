import json

import pytest

from external_agents_eval.manifest import load_task_manifest


def test_load_task_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "dataset_name": "princeton-nlp/SWE-bench_Lite",
                "task_count": 2,
                "task_ids": ["a", "b"],
            }
        )
    )
    manifest = load_task_manifest(path)
    assert manifest.task_ids == ("a", "b")


@pytest.mark.parametrize(
    "task_ids",
    [[], ["a", "a"], ["a", ""]],
)
def test_invalid_task_ids(tmp_path, task_ids):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"task_ids": task_ids}))
    with pytest.raises(ValueError):
        load_task_manifest(path)
