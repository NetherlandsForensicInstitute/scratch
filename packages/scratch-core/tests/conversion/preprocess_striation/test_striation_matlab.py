"""
Tests comparing Python preprocess_striation output with MATLAB PreprocessData.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import CropType, Mark, MarkType
from conversion.preprocess_striation import (
    PreprocessingStriationParams,
    preprocess_striation_mark,
)
from ..helper_functions import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)


MARK_TYPE_MAPPING = {
    "bullet lea striation": MarkType.BULLET_LEA_STRIATION,
    "bullet gea striation": MarkType.BULLET_GEA_STRIATION,
    "breech face impression": MarkType.BREECH_FACE_IMPRESSION,
    "firing pin impression": MarkType.FIRING_PIN_IMPRESSION,
    "firing pin drag": MarkType.FIRING_PIN_DRAG_STRIATION,
    "chamber impression": MarkType.CHAMBER_IMPRESSION,
    "ejector impression": MarkType.EJECTOR_IMPRESSION,
    "extractor impression": MarkType.EXTRACTOR_IMPRESSION,
    "aperture shear striation": MarkType.APERTURE_SHEAR_STRIATION,
    "chamber striation": MarkType.CHAMBER_STRIATION,
    "ejector striation": MarkType.EJECTOR_STRIATION,
    "ejector port striation": MarkType.EJECTOR_PORT_STRIATION,
    "extractor striation": MarkType.EXTRACTOR_STRIATION,
}


def _string_to_mark_type(mark_type_str: str) -> MarkType:
    """Convert MATLAB mark type string to MarkType enum."""
    normalized = mark_type_str.lower().replace("_", " ").replace("mark", "").strip()
    for key, value in MARK_TYPE_MAPPING.items():
        if key in normalized:
            return value
    raise ValueError(f"Unknown mark type: {mark_type_str}")


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB PreprocessData test case."""

    name: str
    input_data: np.ndarray
    input_mask: np.ndarray
    output_data: np.ndarray
    output_mask: np.ndarray

    input_xdim: float = 1.5e-6
    input_ydim: float = 1.5e-6
    mark_type: str = "Bullet LEA striation mark"
    angle_accuracy: float = 90.0
    cutoff_hi: float = 250.0
    cutoff_lo: float = 5.0
    slope_correction: int = 0
    times_median: float = 15.0
    use_mean: bool = True
    shape_noise_removal: bool = True
    show_info: bool = True
    has_mask: bool = False
    output_profile: np.ndarray | None = None
    output_rotation_angle: float | None = None

    @classmethod
    def from_directory(cls, case_dir: Path) -> "MatlabTestCase":
        """Load a test case from a directory."""
        with open(case_dir / "metadata.json") as f:
            meta = json.load(f)

        output_profile = None
        profile_path = case_dir / "output_profile.npy"
        if profile_path.exists():
            output_profile = np.load(profile_path)
            if output_profile.ndim > 1:
                output_profile = output_profile.flatten()

        # Use last two dirs for naming (e.g., "files_with_nans/case_001")
        name = "/".join(case_dir.parts[-2:])

        return cls(
            name=name,
            input_data=np.load(case_dir / "input_data.npy"),
            input_mask=np.load(case_dir / "input_mask.npy"),
            output_data=np.load(case_dir / "output_data.npy"),
            output_mask=np.load(case_dir / "output_mask.npy"),
            output_profile=output_profile,
            **cls._parse_metadata(meta, {f.name for f in fields(cls) if f.init}),
        )

    @staticmethod
    def _parse_metadata(meta: dict, valid_keys: set) -> dict:
        """Parse metadata JSON into dataclass field values."""
        meta["use_mean"] = bool(meta.get("use_mean", 1))
        meta["shape_noise_removal"] = bool(meta.get("shape_noise_removal", 1))
        meta["show_info"] = bool(meta.get("show_info", 1))
        meta["has_mask"] = bool(meta.get("has_mask", 0))
        return {k: v for k, v in meta.items() if k in valid_keys}

    @property
    def is_empty_mask(self) -> bool:
        """Check if mask is empty (all False/zero)."""
        return not np.any(self.input_mask)


def discover_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Discover all test cases in a directory."""
    if not test_cases_dir.exists():
        return []

    cases = []
    for folder in ["files_with_nans", "files_without_nans"]:
        folder_path = test_cases_dir / folder
        if not folder_path.exists():
            continue
        for case_dir in sorted(folder_path.iterdir()):
            if case_dir.is_dir() and (case_dir / "metadata.json").exists():
                try:
                    cases.append(MatlabTestCase.from_directory(case_dir))
                except Exception as e:
                    print(f"Warning: Failed to load test case {case_dir.name}: {e}")
    return cases


@pytest.fixture(scope="module")
def test_cases_dir() -> Path:
    """Get the test cases directory for preprocess_striation tests."""
    return Path(__file__).parent.parent.parent / "resources" / "preprocess_striation"


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
        Path(__file__).parent.parent.parent / "resources" / "preprocess_striation"
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
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float | None]:
    """Run Python preprocess_striation_mark and return the results."""
    scan_image = ScanImage(
        data=test_case.input_data,
        scale_x=test_case.input_xdim,
        scale_y=test_case.input_ydim,
    )

    params = PreprocessingStriationParams(
        highpass_cutoff=test_case.cutoff_hi * 1e-6,
        lowpass_cutoff=test_case.cutoff_lo * 1e-6,
        use_mean=test_case.use_mean,
        angle_accuracy=test_case.angle_accuracy,
    )

    mask = test_case.input_mask.astype(bool) if test_case.has_mask else None
    mark_type = _string_to_mark_type(test_case.mark_type)

    input_mark = Mark(
        scan_image=scan_image,
        mark_type=mark_type,
        crop_type=CropType.RECTANGLE,
    )

    aligned_mark, profile_mark = preprocess_striation_mark(
        mark=input_mark,
        params=params,
        mask=mask,
    )

    aligned_data = aligned_mark.scan_image.data
    profile = profile_mark.scan_image.data.flatten()
    mask_out = aligned_mark.meta_data.get("mask")
    if mask_out is not None:
        mask_out = np.array(mask_out, dtype=bool)
    total_angle = aligned_mark.meta_data.get("total_angle")

    return aligned_data, mask_out, profile, total_angle


class TestPreprocessDataMatlabComparison:
    """Test Python preprocess_data against MATLAB reference outputs."""

    THRESHOLDS = {
        "default": (0.99, 0.003),
        "masked": (0.99, 0.05),
    }

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get thresholds based on test case characteristics."""
        if test_case.has_mask:
            return self.THRESHOLDS["masked"]
        return self.THRESHOLDS["default"]

    def test_processed_output(self, test_case: MatlabTestCase):
        """Test that Python output matches MATLAB reference."""
        python_depth, _, _, _ = run_python_preprocessing(test_case)
        matlab_depth = test_case.output_data

        # Empty mask case
        if test_case.is_empty_mask:
            assert np.all(np.isnan(python_depth)), (
                f"{test_case.name}: empty mask should produce all-NaN output"
            )
            return

        # Shape check (skip for masked cases - MATLAB extracts central region)
        if not test_case.has_mask:
            row_diff = abs(matlab_depth.shape[0] - python_depth.shape[0])
            col_diff = abs(matlab_depth.shape[1] - python_depth.shape[1])
            assert row_diff <= 1 and col_diff <= 1, (
                f"{test_case.name}: shape mismatch "
                f"{python_depth.shape} vs {matlab_depth.shape}"
            )

        # Crop to common shape for value comparisons
        python_depth, matlab_depth = _crop_to_common_shape(
            python_depth, matlab_depth, center_crop=test_case.has_mask
        )
        corr_threshold, std_threshold = self._get_thresholds(test_case)

        # Correlation check
        correlation = _compute_correlation(python_depth, matlab_depth)
        assert correlation > corr_threshold, (
            f"{test_case.name}: correlation {correlation:.4f} < {corr_threshold}"
            f"{' (masked)' if test_case.has_mask else ''}"
        )

        # Difference check
        stats = _compute_difference_stats(python_depth, matlab_depth)
        signal_std = np.nanstd(test_case.output_data)
        relative_std = stats["std"] / signal_std if signal_std > 0 else np.inf
        assert relative_std < std_threshold, (
            f"{test_case.name}: relative_std {relative_std:.4f} > {std_threshold}"
            f"{' (masked)' if test_case.has_mask else ''}"
        )

    def test_profile_correlation(self, test_case: MatlabTestCase):
        """Test that mean/median profile matches MATLAB reference."""
        if test_case.output_profile is None:
            pytest.skip("No profile output in this test case")

        _, _, python_profile, _ = run_python_preprocessing(test_case)

        if python_profile is None:
            pytest.skip("Python implementation did not return profile")

        python_profile = np.asarray(python_profile).flatten()
        matlab_profile = np.asarray(test_case.output_profile).flatten()

        # Align profiles (center crop for masked cases)
        min_len = min(len(python_profile), len(matlab_profile))
        if test_case.has_mask:
            py_start = (len(python_profile) - min_len) // 2
            ml_start = (len(matlab_profile) - min_len) // 2
            python_profile = python_profile[py_start : py_start + min_len]
            matlab_profile = matlab_profile[ml_start : ml_start + min_len]
        else:
            python_profile = python_profile[:min_len]
            matlab_profile = matlab_profile[:min_len]

        correlation = _compute_correlation(
            python_profile.reshape(-1, 1), matlab_profile.reshape(-1, 1)
        )

        corr_threshold, _ = self._get_thresholds(test_case)
        assert correlation > corr_threshold, (
            f"{test_case.name}: profile correlation {correlation:.4f} < {corr_threshold}"
        )

    def test_rotation_angle(self, test_case: MatlabTestCase):
        """Test that rotation angle matches MATLAB reference."""
        if test_case.output_rotation_angle is None:
            pytest.skip("No rotation angle in this test case")

        _, _, _, python_rotation = run_python_preprocessing(test_case)

        if python_rotation is None:
            pytest.skip("Python implementation did not return rotation angle")

        angle_diff = abs(python_rotation - test_case.output_rotation_angle)
        assert angle_diff < 0.1, (
            f"{test_case.name}: rotation angle diff {angle_diff:.4f}° "
            f"(MATLAB: {test_case.output_rotation_angle}, Python: {python_rotation})"
        )

    def test_mask_output(self, test_case: MatlabTestCase):
        """Test that output mask matches MATLAB reference."""
        if not test_case.has_mask:
            pytest.skip("Test case has no mask")

        _, python_mask, _, _ = run_python_preprocessing(test_case)
        assert python_mask is not None, (
            "Python mask should not be None when test case has mask"
        )

        python_mask, matlab_mask = _crop_to_common_shape(
            python_mask, test_case.output_mask, center_crop=True
        )

        python_binary = (python_mask > 0.5).astype(float)
        matlab_binary = (matlab_mask > 0.5).astype(float)

        intersection = np.sum(python_binary * matlab_binary)
        union = np.sum(np.maximum(python_binary, matlab_binary))
        iou = intersection / union if union > 0 else 1.0

        assert iou > 0.95, f"{test_case.name}: mask IoU {iou:.4f} < 0.95"


class TestMatlabTestCaseLoading:
    """Tests for verifying test case loading."""

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
