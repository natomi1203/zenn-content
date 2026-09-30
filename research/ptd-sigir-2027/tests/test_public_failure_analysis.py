import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_public_failure_analysis.py"
SPEC = importlib.util.spec_from_file_location("public_failure_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_positive_bucket_is_complete_and_disjoint():
    assert MODULE._positive_bucket(1) == "1"
    assert MODULE._positive_bucket(2) == "2"
    assert MODULE._positive_bucket(3) == "3_plus"
    assert MODULE._positive_bucket(99) == "3_plus"


def test_bootstrap_is_deterministic():
    values = MODULE.np.asarray([-0.5, 0.0, 0.5], dtype=MODULE.np.float64)
    assert MODULE._bootstrap_interval(values) == MODULE._bootstrap_interval(values)
