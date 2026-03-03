"""Tests for CompareDatasetsNIST: compare Python implementation against MATLAB reference.

Usage:
    pytest test_compare_datasets_nist.py -v

Requires converted test cases in TEST_DATA_DIR (see convert_comparedatasetsnist_tests.py).
"""

from pathlib import Path

import numpy as np
import pytest
from .plot_utils import plot_rotated_squares, plot_side_by_side
from conversion.surface_comparison.pipeline import run_comparison_pipeline
from conversion.surface_comparison.models import ComparisonResult
from .helpers import MatlabTestCase, load_test_case


TEST_DATA_DIR = Path(__file__).parent.parent.parent / "resources" / "cmc"

RTOL_SCALAR = 1e-6
ATOL_SCALAR = 1e-10


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

        ref_scale_x = test_case.reference_map.scale_x
        ref_scale_y = test_case.reference_map.scale_y
        comp_scale_x = test_case.comparison_map.scale_x
        comp_scale_y = test_case.comparison_map.scale_y
        cell_w_px = int(round(test_case.params.cell_size[0] / ref_scale_x))
        cell_h_px = int(round(test_case.params.cell_size[1] / ref_scale_y))

        reference_plot = plot_rotated_squares(
            image=test_case.reference_map.data,
            squares=[
                (
                    (
                        c.center_reference[0] / ref_scale_x,
                        c.center_reference[1] / ref_scale_y,
                    ),
                    (cell_w_px, cell_h_px),
                    0.0,
                )
                for c in self.result.cells
            ],
        )

        comparison_plot = plot_rotated_squares(
            image=test_case.comparison_map.data,
            squares=[
                (
                    (
                        c.center_comparison[0] / comp_scale_x,
                        c.center_comparison[1] / comp_scale_y,
                    ),
                    (cell_w_px, cell_h_px),
                    -c.angle_reference,
                )
                for c in self.result.cells
                if c.center_comparison is not None and c.angle_reference is not None
            ],
        )
        plot_side_by_side(
            img1=reference_plot,
            img2=comparison_plot,
            title1="Reference",
            title2="Comparison",
        )

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
