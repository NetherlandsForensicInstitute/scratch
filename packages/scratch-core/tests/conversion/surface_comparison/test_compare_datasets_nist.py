"""Tests for CompareDatasetsNIST: compare Python implementation against MATLAB reference.

Usage:
    pytest test_compare_datasets_nist.py -v

Requires converted test cases in TEST_DATA_DIR (see convert_comparedatasetsnist_tests.py).
"""

from pathlib import Path

import numpy as np
import pytest

from conversion.surface_comparison.pipeline import run_comparison_pipeline
from conversion.surface_comparison.models import ComparisonResult
from .helpers import MatlabTestCase, load_test_case

# ---- Constants ----

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "resources" / "cmc"

RTOL_SCALAR = 1e-6
ATOL_SCALAR = 1e-10

# ---- Helpers ----


def discover_test_cases() -> list[str]:
    """Find all available test case directories."""
    if not TEST_DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in TEST_DATA_DIR.iterdir() if (d / "metadata.json").exists()
    )


def run_pipeline(test_case: MatlabTestCase) -> ComparisonResult:
    """Execute the comparison pipeline for a test case."""
    return run_comparison_pipeline(
        test_case.reference_map, test_case.comparison_map, test_case.params
    )


# ---- Fixtures ----

test_case_names = discover_test_cases()


@pytest.fixture(params=test_case_names, ids=test_case_names)
def test_case(request: pytest.FixtureRequest) -> MatlabTestCase:
    return load_test_case(TEST_DATA_DIR / request.param)


# ---- Tests ----


@pytest.mark.skipif(
    not test_case_names, reason=f"No test cases found in {TEST_DATA_DIR}"
)
class TestCompareDatasetsNIST:
    """Test CompareDatasetsNIST Python implementation against MATLAB reference."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self, test_case: MatlabTestCase) -> None:
        self.result = run_pipeline(test_case)

    def test_res_scalar_fields(self, test_case: MatlabTestCase) -> None:
        """Test that Res scalar output fields match MATLAB."""
        if not test_case.expected_results:
            pytest.skip("No comparable scalar fields in Res output")

        failures = []
        for field, expected in test_case.expected_results.items():
            actual = getattr(self.result, field)

            if np.isnan(expected):
                if not np.isnan(actual):
                    failures.append(f"  {field}: expected NaN, got {actual}")
            elif not np.isclose(actual, expected, rtol=RTOL_SCALAR, atol=ATOL_SCALAR):
                failures.append(
                    f"  {field}: expected {expected}, got {actual}, diff={abs(actual - expected):.2e}"
                )

        if failures:
            pytest.fail("Scalar field mismatches:\n" + "\n".join(failures))
