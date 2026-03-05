"""Tests for CompareDatasetsNIST: compare simone (MATLAB-faithful) pipeline against MATLAB reference.

Usage:
    pytest test_compare_datasets_nist_old_simone_way.py -v

Requires converted test cases in TEST_DATA_DIR (see convert_comparedatasetsnist_tests.py).
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest
from matplotlib import pyplot as plt

from conversion.surface_comparison.cell_registration import register_cells
from conversion.surface_comparison.cmc_classification import (
    classify_congruent_cells,
)
from conversion.surface_comparison.models import (
    ComparisonParams,
    ComparisonResult,
)
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
    """Execute the simone comparison pipeline for a test case."""
    params = ComparisonParams(
        cell_size=test_case.params.cell_size.copy(),
        minimum_fill_fraction=test_case.params.minimum_fill_fraction,
        correlation_threshold=test_case.params.correlation_threshold,
        angle_threshold=test_case.params.angle_threshold,
        position_threshold=test_case.params.position_threshold,
        search_angle_min=test_case.params.search_angle_min,
        search_angle_max=test_case.params.search_angle_max,
        search_angle_step=test_case.params.search_angle_step,
    )
    result = ComparisonResult()
    result.cells = register_cells(
        test_case.reference_map, test_case.comparison_map, params
    )
    classify_congruent_cells(result, params, test_case.reference_map.global_center)
    result.update_summary()
    return result


def plot_rotated_squares(
    image: np.ndarray,
    squares: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
) -> np.ndarray:
    """
    Plots rotated squares on a float-based image.

    Args:
        image: Input array (e.g., magnitude 1e-6).
        squares: List of ((cx, cy), (w, h), angle_deg).

    Returns:
        Annotated image in BGR format (uint8).
    """
    img_min, img_max = np.nanmin(image), np.nanmax(image)
    if img_max - img_min != 0:
        norm_img = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        norm_img = np.zeros(image.shape, dtype=np.uint8)

    color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

    for rect in squares:
        box_points = cv2.boxPoints(rect)
        box_points = np.int64(box_points)
        cv2.drawContours(color_img, [box_points], 0, (0, 255, 0), 2)
    return color_img


def plot_side_by_side(
    img1: np.ndarray, img2: np.ndarray, title1: str = "Image 1", title2: str = "Image 2"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


test_case_names = discover_test_cases()


@pytest.fixture(params=test_case_names, ids=test_case_names)
def test_case(request: pytest.FixtureRequest) -> MatlabTestCase:
    return load_test_case(TEST_DATA_DIR / request.param)


@pytest.mark.skipif(
    not test_case_names, reason=f"No test cases found in {TEST_DATA_DIR}"
)
class TestCompareDatasetsNISTSimone:
    """Test simone (MATLAB-faithful) pipeline against MATLAB reference."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self, test_case: MatlabTestCase) -> None:
        self.result = run_pipeline(test_case)

    def test_res_scalar_fields(
        self, test_case: MatlabTestCase, plot: bool = True
    ) -> None:
        """Test that Res scalar output fields match MATLAB."""
        if not test_case.expected_results:
            pytest.skip("No comparable scalar fields in Res output")

        ref_scale_x = test_case.reference_map.scale_x
        ref_scale_y = test_case.reference_map.scale_y
        comp_scale_x = test_case.comparison_map.scale_x
        comp_scale_y = test_case.comparison_map.scale_y
        cell_w_px = int(round(test_case.params.cell_size[0] / ref_scale_x))
        cell_h_px = int(round(test_case.params.cell_size[1] / ref_scale_y))

        if plot:
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
                        -np.degrees(c.registration_angle),
                    )
                    for c in self.result.cells
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
