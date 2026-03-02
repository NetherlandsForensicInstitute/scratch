"""Tests for CompareDatasetsNIST: compare Python implementation against MATLAB reference.

Usage:
    pytest test_compare_datasets_nist.py -v

Requires converted test cases in TEST_DATA_DIR (see convert_comparedatasetsnist_tests.py).
"""

from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from conversion.surface_comparison.pipeline import run_comparison_pipeline
from conversion.surface_comparison.models import ComparisonResult
from .helpers import MatlabTestCase, load_test_case

import cv2
from typing import List, Tuple

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
    # 1. Normalize the 1e-6 data to 0-255 range for visualization
    img_min, img_max = np.nanmin(image), np.nanmax(image)
    if img_max - img_min != 0:
        norm_img = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        norm_img = np.zeros(image.shape, dtype=np.uint8)

    # Convert grayscale to BGR to allow colored drawings
    color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

    # 2. Draw each rotated square
    for rect in squares:
        # cv2.boxPoints returns the 4 corners of the rotated rect
        box_points = cv2.boxPoints(rect)
        # Coordinates must be integers for drawing functions
        box_points = np.int64(box_points)

        # Draw the contour (Green, thickness 2)
        cv2.drawContours(color_img, [box_points], 0, (0, 255, 0), 2)
    return color_img


def plot_side_by_side(
    img1: np.ndarray, img2: np.ndarray, title1: str = "Image 1", title2: str = "Image 2"
) -> None:
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot first image
    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis("off")  # Hide grid/axes

    # Plot second image
    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


####

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

        reference_plot = plot_rotated_squares(
            image=test_case.reference_map.data,
            squares=[
                (
                    (c.center_reference[0] / 3.5e-06, c.center_reference[1] / 3.5e-06),
                    (c.cell_data.shape[1], c.cell_data.shape[0]),
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
                        c.center_comparison[0] / 3.5e-06,
                        c.center_comparison[1] / 3.5e-06,
                    ),
                    (c.cell_data.shape[1], c.cell_data.shape[0]),
                    -c.angle_reference,
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
