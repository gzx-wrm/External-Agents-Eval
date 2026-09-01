import json

import pytest

from external_agents_eval.benchmarks.deepswe.config import DEFAULT_MANIFEST
from external_agents_eval.benchmarks.deepswe.manifest import (
    discover_tasks,
    load_manifest,
    population_fingerprint,
    select_task_ids,
    write_manifest,
)


def _make_tasks(tmp_path, count=12):
    root = tmp_path / "tasks"
    for index in range(count):
        task = root / f"task-{index:03d}"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            'environment_mode = "separate"\ndocker_image = "example-v1.1"\n'
        )
        (task / "instruction.md").write_text(f"Solve task {index}\n")
    return root


def test_packaged_default_manifest_is_fixed_random_60():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert manifest.sample_seed == 20260901
    assert manifest.sample_size == len(manifest.task_ids) == 60
    assert manifest.population_count == 113


def test_fixed_sampling_is_independent_of_input_order(tmp_path):
    tasks = discover_tasks(_make_tasks(tmp_path))
    selected = select_task_ids(tasks, sample_size=6, seed=20260901)
    reversed_selected = select_task_ids(list(reversed(tasks)), sample_size=6, seed=20260901)
    assert selected == reversed_selected
    assert len(selected) == len(set(selected)) == 6


def test_population_fingerprint_covers_task_definition(tmp_path):
    tasks = discover_tasks(_make_tasks(tmp_path, count=2))
    before = population_fingerprint(tasks)
    (tasks[0] / "instruction.md").write_text("Changed instruction\n")
    assert population_fingerprint(tasks) != before


def test_manifest_is_immutable_once_written(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, {"task_ids": ["a"]})
    write_manifest(path, {"task_ids": ["a"]})
    with pytest.raises(ValueError, match="different contents"):
        write_manifest(path, {"task_ids": ["b"]})
    assert json.loads(path.read_text()) == {"task_ids": ["a"]}


def test_load_manifest_rejects_task_list_fingerprint_mismatch(tmp_path):
    source = {
        "schema_version": 1,
        "benchmark": "deepswe",
        "benchmark_revision": "1.1",
        "source": {
            "repository": "https://github.com/datacurve-ai/deep-swe.git",
            "commit": "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea",
        },
        "sampling": {
            "algorithm": "sorted-task-ids/python-random-shuffle-v1",
            "seed": 1,
            "size": 1,
        },
        "population": {"task_count": 113, "fingerprint_sha256": "a" * 64},
        "task_count": 1,
        "task_ids": ["changed"],
        "selection_fingerprint_sha256": "b" * 64,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="selection fingerprint"):
        load_manifest(path)
