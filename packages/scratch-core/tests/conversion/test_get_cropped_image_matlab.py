"""
Tests comparing Python get_cropped_image output with MATLAB GetPreviewImageForCropping.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from container_models.base import ScanMap2DArray, MaskArray
from container_models.scan_image import ScanImage
from conversion.get_cropped_image import get_cropped_image
from conversion.leveling import SurfaceTerms
from .helper_functions import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)

LEVEL_METHOD_MAP = {
    "none": SurfaceTerms(0),
    "plane": SurfaceTerms.PLANE,
    "sphere": SurfaceTerms.SPHERE,
}

FILTER_METHOD_MAP = {
    "r0": 0,
    "r1": 1,
    "r2": 2,
}


def to_surface_terms(level_method: str) -> SurfaceTerms:
    """Convert level method to SurfaceTerms."""
    key = level_method.strip().lower()
    if key not in LEVEL_METHOD_MAP:
        raise ValueError(f"Unknown level method: {level_method}")
    return LEVEL_METHOD_MAP[key]


def to_regression_order(filter_method: str) -> int:
    """Convert filter method to regression order."""
    key = filter_method.strip().lower()
    if key not in FILTER_METHOD_MAP:
        raise ValueError(f"Unknown filter method: {filter_method}")
    return FILTER_METHOD_MAP[key]


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB test case."""

    name: str
    # Input
    input_depth_data: ScanMap2DArray
    input_xdim: float
    input_ydim: float
    input_mask: MaskArray
    # Processing options
    terms: SurfaceTerms
    regression_order: int
    cutoff_length: float
    sampling: float | None
    # Expected output
    output_depth_data: ScanMap2DArray
    output_xdim: float
    output_ydim: float
    has_mask: bool

    @classmethod
    def from_directory(cls, case_dir: Path) -> "MatlabTestCase":
        """Load a test case from a directory."""
        with open(case_dir / "metadata.json") as f:
            meta = json.load(f)

        return cls(
            name=case_dir.name,
            input_depth_data=np.load(case_dir / "input_data.npy"),
            input_xdim=float(meta.get("input_xdim", 3.5e-6)),
            input_ydim=float(meta.get("input_ydim", 3.5e-6)),
            input_mask=np.load(case_dir / "input_mask.npy"),
            terms=to_surface_terms(meta.get("level_method", "Plane")),
            regression_order=to_regression_order(meta.get("filter_method", "R0")),
            cutoff_length=float(meta.get("cutoff_hi", 250.0)),
            sampling=float(meta["sampling"])
            if meta.get("sampling") is not None
            else None,
            output_depth_data=np.load(case_dir / "output_data.npy"),
            output_xdim=float(
                meta.get("output_xdim") or meta.get("input_xdim", 3.5e-6)
            ),
            output_ydim=float(
                meta.get("output_ydim") or meta.get("input_ydim", 3.5e-6)
            ),
            has_mask=meta.get("has_mask", False),
        )

    @property
    def involves_resampling(self) -> bool:
        """Check if test case involves resampling."""
        if self.sampling is not None:
            return True
        return self.input_xdim / 4e-6 < 1

    @property
    def is_empty_mask(self) -> bool:
        """Check if mask is empty (all False/zero)."""
        return not np.any(self.input_mask)


def discover_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Discover all test cases in a directory."""
    if not test_cases_dir.exists():
        return []
    return [
        MatlabTestCase.from_directory(d)
        for d in sorted(test_cases_dir.iterdir())
        if d.is_dir() and (d / "metadata.json").exists()
    ]


@pytest.fixture(scope="module")
def test_cases_dir(baseline_images_dir) -> Path:
    return baseline_images_dir / "get_cropped_image"


@pytest.fixture(scope="module")
def all_test_cases(test_cases_dir) -> list[MatlabTestCase]:
    """Load all MATLAB test cases."""
    cases = discover_test_cases(test_cases_dir)
    if not cases:
        pytest.skip(f"No test cases found in {test_cases_dir}")
    return cases


def pytest_generate_tests(metafunc):
    """Generate test cases dynamically."""
    if "test_case_name" not in metafunc.fixturenames:
        return

    test_cases_dir = (
        Path(__file__).parent.parent
        / "resources"
        / "baseline_images"
        / "get_cropped_image"
    )
    cases = discover_test_cases(test_cases_dir)
    metafunc.parametrize("test_case_name", [c.name for c in cases])  # <-- created here


@pytest.fixture
def test_case(
    test_case_name: str, all_test_cases: list[MatlabTestCase]
) -> MatlabTestCase:
    """Get individual test case by name."""
    for case in all_test_cases:
        if case.name == test_case_name:
            return case
    pytest.skip(f"Test case {test_case_name} not found")


def run_python_preprocessing(test_case: MatlabTestCase) -> np.ndarray:
    """Run Python get_cropped_image and return the result."""
    scan_image = ScanImage(
        data=test_case.input_depth_data,
        scale_x=test_case.input_xdim,
        scale_y=test_case.input_ydim,
    )

    resampling_factors = (
        (test_case.sampling, test_case.sampling) if test_case.sampling else None
    )
    return get_cropped_image(
        scan_image=scan_image,
        mask=test_case.input_mask.astype(bool),
        terms=test_case.terms,
        cutoff_length=test_case.cutoff_length * 1e-6,
        regression_order=test_case.regression_order,
        resampling_factors=resampling_factors,
        crop=False,
    )


class TestGetCroppedImageMatlabComparison:
    """Test Python get_cropped_image against MATLAB reference outputs."""

    THRESHOLDS = {
        "default": (0.999, 0.01),
        "resampled": (0.95, 0.25),
        "masked": (0.95, 0.20),
        "resampled_masked": (0.90, 0.30),
    }

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get thresholds based on test case characteristics."""
        if test_case.involves_resampling and test_case.has_mask:
            return self.THRESHOLDS["resampled_masked"]
        if test_case.has_mask:
            return self.THRESHOLDS["masked"]
        if test_case.involves_resampling:
            return self.THRESHOLDS["resampled"]
        return self.THRESHOLDS["default"]

    def test_matlab_comparison(self, test_case: MatlabTestCase):
        """Test that Python output matches MATLAB reference."""
        python_result = run_python_preprocessing(test_case)
        matlab_result = test_case.output_depth_data

        # Empty mask case
        if test_case.is_empty_mask:
            assert np.all(np.isnan(python_result)), (
                f"{test_case.name}: empty mask should produce all-NaN output"
            )
            return

        # Shape check
        max_diff = 2 if test_case.involves_resampling else 1
        row_diff = abs(matlab_result.shape[0] - python_result.shape[0])
        col_diff = abs(matlab_result.shape[1] - python_result.shape[1])
        assert row_diff <= max_diff and col_diff <= max_diff, (
            f"{test_case.name}: shape mismatch "
            f"{python_result.shape} vs {matlab_result.shape}"
        )

        # Crop to common shape for value comparisons
        python_result, matlab_result = _crop_to_common_shape(
            python_result, matlab_result
        )
        corr_threshold, std_threshold = self._get_thresholds(test_case)

        # Correlation check
        correlation = _compute_correlation(python_result, matlab_result)
        assert correlation > corr_threshold, (
            f"{test_case.name}: correlation {correlation:.4f} < {corr_threshold}"
        )

        # Difference check
        stats = _compute_difference_stats(python_result, matlab_result)
        signal_std = np.nanstd(test_case.output_depth_data)
        relative_std = stats["std"] / signal_std if signal_std > 0 else np.inf
        assert relative_std < std_threshold, (
            f"{test_case.name}: relative_std {relative_std:.4f} > {std_threshold}"
        )
