from pathlib import Path

from external_agents_eval.benchmarks.base import BenchmarkSpec

DEEPSWE_REPOSITORY = "https://github.com/datacurve-ai/deep-swe.git"
DEEPSWE_COMMIT = "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea"
DEEPSWE_REVISION = "1.1"
DEEPSWE_TASK_COUNT = 113
DEFAULT_SAMPLE_SIZE = 60
DEFAULT_SAMPLE_SEED = 20260901
DEFAULT_CHECKOUT_DIR = Path(".benchmarks/deep-swe")
DEFAULT_MANIFEST = Path(__file__).with_name("data") / "deepswe_v1.1_random60_seed20260901.json"
DEFAULT_JOBS_DIR = Path("results/deepswe")
PIER_REQUIREMENT = "datacurve-pier==0.3.1"

DEEPSWE = BenchmarkSpec(
    benchmark_id="deepswe",
    revision=DEEPSWE_REVISION,
    repository_url=DEEPSWE_REPOSITORY,
    workdir="/app",
)
