"""
Tests comparing Python rotate_crop_image output with MATLAB RotateCropImage.
"""

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pytest

from container_models.base import ScanMap2DArray, MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import CropType, CropInfo
from conversion.rotate import get_rotation_angle, rotate_crop_and_mask_image_by_crop
from .helper_functions import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB RotateCropImage test case."""

    name: str
    input_data: ScanMap2DArray
    input_mask: MaskArray
    output_data: ScanMap2DArray
    output_mask: MaskArray

    input_xdim: float = 3.5e-6
    input_ydim: float = 3.5e-6
    rotation_angle: float = 0.0
    crop_type: str = "rectangle"
    crop_corners: np.ndarray | None = None
    crop_foreground: bool = True
    times_median: float = 15.0
    has_holes: bool = False

    @classmethod
    def from_directory(cls, case_dir: Path) -> "MatlabTestCase":
        """Load a test case from a directory."""
        with open(case_dir / "metadata.json") as f:
            meta = json.load(f)

        # Load crop corners if present
        crop_corners = None
        crop_corners_path = case_dir / "crop_corners.npy"
        if crop_corners_path.exists():
            crop_corners = np.load(crop_corners_path)

        return cls(
            name=case_dir.name,
            input_data=np.load(case_dir / "input_data.npy"),
            input_mask=np.load(case_dir / "input_mask.npy"),
            output_data=np.load(case_dir / "output_data.npy"),
            output_mask=np.load(case_dir / "output_mask.npy"),
            crop_corners=crop_corners,
            **cls._parse_metadata(meta, {f.name for f in fields(cls) if f.init}),
        )

    @staticmethod
    def _parse_metadata(meta: dict, valid_keys: set) -> dict:
        """Parse metadata JSON into dataclass field values."""
        return {k: v for k, v in meta.items() if k in valid_keys}

    @property
    def involves_rotation(self) -> bool:
        """Check if test case involves rotation."""
        if self.rotation_angle != 0:
            return True
        return "rotation" in self.name.lower()

    @property
    def expected_rotation_angle(self) -> float | None:
        """Extract expected rotation angle from test name if present."""
        match = re.findall(r"rotation_?(neg)?(\d+)deg", self.name.lower())
        if not match:
            return None
        angle = float(match[0][1])
        if match[0][0]:  # neg prefix
            angle = -angle
        return angle

    def to_crop_info(self) -> tuple[CropInfo]:
        """Convert MATLAB crop_info to Python CropInfo list."""
        if self.crop_corners is None:
            return (
                CropInfo(
                    crop_type=CropType.CIRCLE,
                    data={"center": np.array([23, 30]), "radius": 2.4},
                    is_foreground=True,
                ),
            )

        crop_type = (
            CropType.RECTANGLE
            if self.crop_type.lower() == "rectangle"
            else CropType.POLYGON
        )
        return (
            CropInfo(
                crop_type=crop_type,
                data={"corner": self.crop_corners},
                is_foreground=self.crop_foreground,
            ),
        )


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
    return baseline_images_dir / "rotate_crop_image"


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
        / "rotate_crop_image"
    )
    cases = discover_test_cases(test_cases_dir)
    metafunc.parametrize("test_case_name", [c.name for c in cases])


@pytest.fixture
def test_case(
    test_case_name: str, all_test_cases: list[MatlabTestCase]
) -> MatlabTestCase:
    """Get individual test case by name."""
    for case in all_test_cases:
        if case.name == test_case_name:
            return case
    pytest.skip(f"Test case {test_case_name} not found")


def run_python_preprocessing(
    test_case: MatlabTestCase,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Python rotate_crop_image and return the result."""
    scan_image = ScanImage(
        data=test_case.input_data.copy(),
        scale_x=test_case.input_xdim,
        scale_y=test_case.input_ydim,
    )

    data_out, mask_out = rotate_crop_and_mask_image_by_crop(
        scan_image=scan_image,
        mask=test_case.input_mask.copy(),
        crop_infos=test_case.to_crop_info(),
        rotation_angle=test_case.rotation_angle,
        times_median=test_case.times_median,
    )
    return data_out.data, mask_out


class TestRotateCropImageMatlabComparison:
    """Test Python rotate_crop_image against MATLAB reference outputs."""

    THRESHOLDS = {
        "default": (0.99, 0.01),
        "rotated": (0.95, 0.25),
        "holes": (0.95, 0.20),
        "rotated_holes": (0.90, 0.30),
    }

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get thresholds based on test case characteristics."""
        if test_case.involves_rotation and test_case.has_holes:
            return self.THRESHOLDS["rotated_holes"]
        if test_case.has_holes:
            return self.THRESHOLDS["holes"]
        if test_case.involves_rotation:
            return self.THRESHOLDS["rotated"]
        return self.THRESHOLDS["default"]

    def test_matlab_comparison(self, test_case: MatlabTestCase):
        """Test that Python output matches MATLAB reference."""
        python_data, python_mask = run_python_preprocessing(test_case)
        matlab_data = test_case.output_data

        # Shape check
        max_diff = 5 if test_case.involves_rotation else 2
        row_diff = abs(matlab_data.shape[0] - python_data.shape[0])
        col_diff = abs(matlab_data.shape[1] - python_data.shape[1])
        assert row_diff <= max_diff and col_diff <= max_diff, (
            f"{test_case.name}: shape mismatch "
            f"{python_data.shape} vs {matlab_data.shape}"
        )

        # Crop to common shape for value comparisons
        python_data, matlab_data = _crop_to_common_shape(python_data, matlab_data)

        # Check valid pixels
        valid_mask = ~np.isnan(python_data) & ~np.isnan(matlab_data)
        if np.sum(valid_mask) < 10:
            pytest.skip("Too few valid pixels for comparison")

        corr_threshold, std_threshold = self._get_thresholds(test_case)

        # Correlation check
        correlation = _compute_correlation(python_data, matlab_data)
        assert round(float(correlation), 2) >= corr_threshold, (
            f"{test_case.name}: correlation {correlation:.4f} < {corr_threshold}"
        )

        # Difference check
        stats = _compute_difference_stats(python_data, matlab_data)
        python_result_std = np.nanstd(python_data)
        matlab_result_std = np.nanstd(matlab_data)
        combined_std = np.sqrt(python_result_std**2 + matlab_result_std**2)
        relative_std = stats["std"] / combined_std if combined_std > 0 else np.inf
        assert relative_std < std_threshold, (
            f"{test_case.name}: relative_std {relative_std:.4f} > {std_threshold}"
        )

    def test_rotation_angle_calculation(self, test_case: MatlabTestCase):
        """Test that rotation angle is calculated correctly from corners."""
        if test_case.rotation_angle != 0:
            pytest.skip("Rotation angle was pre-specified")

        expected = test_case.expected_rotation_angle
        if expected is None:
            pytest.skip("No expected angle in test name")

        calculated_angle = get_rotation_angle(
            crop_infos=test_case.to_crop_info(), rotation_angle=0.0
        )

        assert abs(calculated_angle - expected) < 1.0, (
            f"{test_case.name}: calculated angle {calculated_angle:.2f} != {expected}"
        )


class TestMatlabTestCaseLoading:
    """Tests for verifying test case loading works correctly."""

    def test_discover_test_cases(self, test_cases_dir):
        """Test that test cases can be discovered."""
        cases = discover_test_cases(test_cases_dir)
        print(f"Found {len(cases)} test cases in {test_cases_dir}")

    def test_load_test_case_structure(self, all_test_cases: list[MatlabTestCase]):
        """Test that loaded test cases have valid structure."""
        for case in all_test_cases:
            assert case.input_data is not None
            assert case.input_data.ndim == 2
            assert case.input_xdim > 0
            assert case.input_ydim > 0
            assert case.input_mask is not None
            assert case.input_mask.shape == case.input_data.shape
            assert case.output_data is not None
            assert case.output_data.ndim == 2
            assert case.crop_type.lower() in ("rectangle", "polygon", "circle")

    def test_print_test_case_summary(self, all_test_cases: list[MatlabTestCase]):
        """Print summary of all test cases for debugging."""
        print("\n=== Test Case Summary ===")
        for case in all_test_cases:
            print(f"\n{case.name}:")
            print(f"  Input shape: {case.input_data.shape}")
            print(f"  Output shape: {case.output_data.shape}")
            print(f"  Rotation angle: {case.rotation_angle}")
            print(f"  Crop type: {case.crop_type}")
            print(f"  Has holes: {case.has_holes}")
