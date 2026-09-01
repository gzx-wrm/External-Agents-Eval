import json

from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest
from external_agents_eval.benchmarks.deepswe.report import collect_summary


def _manifest(tmp_path):
    return DeepSWEManifest(
        path=tmp_path / "manifest.json",
        task_ids=("passed", "failed", "pending"),
        sample_seed=1,
        sample_size=3,
        population_count=113,
        population_fingerprint="a" * 64,
        raw={},
    )


def _write_result(job_dir, trial, task, reward, *, filename="result.json", namespaced=True):
    path = job_dir / trial
    path.mkdir(parents=True)
    task_name = f"datacurve/{task}" if namespaced else task
    (path / filename).write_text(
        json.dumps(
            {
                "task_name": task_name,
                "finished_at": "2026-09-01T00:00:00Z",
                "verifier_result": {"rewards": {"reward": reward}},
            }
        )
    )


def test_collect_deepswe_summary(tmp_path):
    job_dir = tmp_path / "job"
    _write_result(job_dir, "passed__one", "passed", 1)
    _write_result(
        job_dir,
        "failed__one",
        "failed",
        0,
        filename="results.json",
        namespaced=False,
    )
    summary = collect_summary(job_dir, _manifest(tmp_path))
    assert summary["counts"] == {"failed": 1, "passed": 1, "pending": 1}
    assert summary["evaluated_count"] == 2
    assert summary["pass_rate"] is None
    assert summary["evaluated_pass_rate"] == 0.5
