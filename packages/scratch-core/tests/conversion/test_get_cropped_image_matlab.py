"""
Tests comparing Python get_cropped_image output with MATLAB GetPreviewImageForCropping.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pytest

from container_models.base import FloatArray2D, BinaryMask
from container_models.scan_image import ScanImage
from conversion.leveling import SurfaceTerms
from conversion.resample import get_scaling_factors
from parsers import save_x3p, parse_to_x3p
from preprocessors.controller import apply_changes_on_scan_image

from preprocessors.schemas import EditImage
from utils.constants import RegressionOrder
from .helper_functions import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)

LEVEL_METHOD_MAP = {
    "none": SurfaceTerms.NONE,
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
    input_data: FloatArray2D
    input_mask: BinaryMask
    output_data: FloatArray2D

    input_xdim: float = 3.5e-6
    input_ydim: float = 3.5e-6
    terms: SurfaceTerms = field(default_factory=lambda: to_surface_terms("Plane"))
    regression_order: int = field(default_factory=lambda: to_regression_order("R0"))
    cutoff_length: float = 250.0
    resampling_factor: float | None = None
    output_xdim: float | None = None
    output_ydim: float | None = None
    has_mask: bool = False

    def __post_init__(self) -> None:
        """Set output dimensions to input dimensions if not specified."""
        if self.output_xdim is None:
            self.output_xdim = self.input_xdim
        if self.output_ydim is None:
            self.output_ydim = self.input_ydim

    @classmethod
    def from_directory(cls, case_dir: Path) -> "MatlabTestCase":
        """Load a test case from a directory."""
        with open(case_dir / "metadata.json") as f:
            meta = json.load(f)

        return cls(
            name=case_dir.name,
            input_data=np.load(case_dir / "input_data.npy"),
            input_mask=np.load(case_dir / "input_mask.npy"),
            output_data=np.load(case_dir / "output_data.npy"),
            **cls._parse_metadata(meta, {f.name for f in fields(cls) if f.init}),
        )

    @staticmethod
    def _parse_metadata(meta: dict, valid_keys: set) -> dict:
        """Parse metadata JSON into dataclass field values."""
        meta["terms"] = to_surface_terms(meta["level_method"])
        meta["regression_order"] = to_regression_order(meta["filter_method"])
        meta["cutoff_length"] = meta["cutoff_hi"]
        meta["resampling_factor"] = meta["sampling"]
        meta["output_xdim"] = meta["output_xdim"] or meta["input_xdim"]
        meta["output_ydim"] = meta["output_ydim"] or meta["input_ydim"]
        return {k: v for k, v in meta.items() if k in valid_keys}

    @property
    def involves_resampling(self) -> bool:
        """Check if test case involves resampling."""
        if self.resampling_factor is not None and self.resampling_factor != 1.0:
            return True
        if not self.output_xdim or not self.output_ydim:
            return False
        return (
            abs(self.output_xdim - self.input_xdim) > 1e-12
            or abs(self.output_ydim - self.input_ydim) > 1e-12
        )

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


@pytest.fixture
def tmp_path(tmp_path: Path):
    return tmp_path


def run_python_preprocessing(
    test_case: MatlabTestCase,
    tmp_path: Path,
) -> FloatArray2D:
    """Run Python get_cropped_image and return the result."""
    scan_image = ScanImage(
        data=test_case.input_data,
        scale_x=test_case.input_xdim,
        scale_y=test_case.input_ydim,
    )

    resampling_factors = (
        (test_case.resampling_factor, test_case.resampling_factor)
        if test_case.resampling_factor
        else get_scaling_factors(
            scales=(scan_image.scale_x, scan_image.scale_y), target_scale=4e-6
        )
    )
    scan_file = tmp_path / "scan.x3p"
    save_x3p(output_path=scan_file, x3p=parse_to_x3p(scan_image).unwrap())
    mask_tuple = tuple(
        tuple(bool(x) for x in row) for row in test_case.input_mask.astype(bool)
    )
    params = EditImage(
        project_name="test",
        scan_file=tmp_path / "scan.x3p",
        mask=mask_tuple,
        cutoff_length=test_case.cutoff_length * 1e-6,
        resampling_factor=resampling_factors[0],
        terms=test_case.terms,
        regression_order=RegressionOrder(test_case.regression_order),
        crop=False,
        step_size_x=1,
        step_size_y=1,
    )

    return apply_changes_on_scan_image(
        scan_image=scan_image,
        edit_image_params=params,
        mask=np.zeros(scan_image.data.shape, dtype=np.bool_),
    ).data


class TestGetCroppedImageMatlabComparison:
    """Test Python get_cropped_image against MATLAB reference outputs."""

    THRESHOLDS = {
        "default": (0.999, 0.01),
        "resampled": (0.97, 0.18),
        "resampled_masked": (0.98, 0.13),
    }

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get thresholds based on test case characteristics."""
        if test_case.involves_resampling and test_case.has_mask:
            return self.THRESHOLDS["resampled_masked"]
        if test_case.involves_resampling:
            return self.THRESHOLDS["resampled"]
        return self.THRESHOLDS["default"]

    def test_matlab_comparison(self, test_case: MatlabTestCase, tmp_path: Path):
        """Test that Python output matches MATLAB reference."""
        python_result = run_python_preprocessing(test_case, tmp_path)
        matlab_result = test_case.output_data

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
        python_result_std = np.nanstd(python_result)
        matlab_result_std = np.nanstd(matlab_result)
        combined_std = np.sqrt(python_result_std**2 + matlab_result_std**2)
        relative_std = stats["std"] / combined_std if combined_std > 0 else np.inf
        assert relative_std < std_threshold, (
            f"{test_case.name}: relative_std {relative_std:.4f} > {std_threshold}"
        )
