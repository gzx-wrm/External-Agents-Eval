from pathlib import Path

from external_agents_eval.benchmarks.base import BenchmarkSpec

SWEBENCH = BenchmarkSpec(
    benchmark_id="swebench",
    revision="lite",
    repository_url="https://github.com/swe-bench/SWE-bench",
    workdir="/testbed",
)

DEFAULT_MANIFEST = Path(
    "/Users/gzx/code/python/HGM/offline_tools/manifests/lite_filtered_random50_seed20260818.json"
)
DEFAULT_RESULTS_DIR = Path("results_lite_filtered_random50_seed20260818")
DEFAULT_SWEBENCH_RUNNER = Path(
    "/Users/gzx/code/python/HGM/swe_bench/SWE-bench/swebench/harness/run_evaluation.py"
)
