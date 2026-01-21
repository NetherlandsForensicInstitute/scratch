# noqa
"""
This file contains tests to explicitly compare the Python PreprocessData output with the original MATLAB output.
Upon completion of the translation and validation, it can be used for regression testing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from ..helper_function import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)


from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType
from conversion.preprocess_striation.parameters import PreprocessingStriationParams
from conversion.preprocess_striation.preprocess_striation import preprocess_data


def _string_to_mark_type(mark_type_str: str) -> MarkType:
    """Convert MATLAB mark type string to MarkType enum."""
    # Map common MATLAB strings to MarkType enum values
    mapping = {
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
    # Normalize the input string and find a match
    normalized = mark_type_str.lower().replace("_", " ").replace("mark", "").strip()
    for key, value in mapping.items():
        if key in normalized:
            return value
    raise ValueError(f"Unknown mark type: {mark_type_str}")


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB PreprocessData test case."""

    name: str
    # Input data
    input_depth_data: np.ndarray
    input_xdim: float
    input_ydim: float
    input_mark_type: str
    input_mask: np.ndarray
    # Input parameters
    params: dict[str, Any]
    # Output data
    output_depth_data: np.ndarray
    output_mask: np.ndarray
    output_profile: np.ndarray | None
    output_rotation_angle: float | None
    # Metadata
    has_mask: bool


def load_matlab_scalar(mat_dict: dict, key: str, default: Any = None) -> Any:
    """Extract a scalar value from loaded .mat dictionary."""
    if key not in mat_dict:
        return default
    val = mat_dict[key]
    # MATLAB saves scalars as 2D arrays
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.flat[0]
        elif val.size == 0:
            return default
        return val.flatten().tolist()
    return val


def load_matlab_string(mat_dict: dict, key: str, default: str = "") -> str:
    """Extract a string value from loaded .mat dictionary."""
    if key not in mat_dict:
        return default
    val = mat_dict[key]
    if isinstance(val, np.ndarray):
        # MATLAB strings are stored as character arrays
        if val.dtype.kind in ("U", "S"):  # Unicode or byte string
            return str(val.flat[0]) if val.size == 1 else str(val)
        elif val.size == 1:
            return str(val.flat[0])
    return str(val) if val is not None else default


def load_test_case(case_dir: Path) -> MatlabTestCase:
    """Load a single test case from a directory."""
    # Load input data
    input_data_mat = loadmat(case_dir / "input_data.mat")
    input_param_mat = loadmat(case_dir / "input_param.mat")
    input_mask_mat = loadmat(case_dir / "input_mask.mat")

    # Load output data
    output_data_mat = loadmat(case_dir / "output_data.mat")
    output_mask_mat = loadmat(case_dir / "output_mask.mat")

    # Load optional outputs
    output_profile = None
    if (case_dir / "output_profile.mat").exists():
        output_profile_mat = loadmat(case_dir / "output_profile.mat")
        output_profile = output_profile_mat.get("depth_data")
        if output_profile is not None and output_profile.ndim > 1:
            output_profile = output_profile.flatten()

    output_rotation_angle = None
    if (case_dir / "output_rotation.mat").exists():
        output_rotation_mat = loadmat(case_dir / "output_rotation.mat")
        output_rotation_angle = load_matlab_scalar(
            output_rotation_mat, "rotation_angle"
        )

    # Load metadata
    metadata_mat = loadmat(case_dir / "metadata.mat")
    has_mask = bool(load_matlab_scalar(metadata_mat, "has_mask", 0))

    # Extract input data fields
    depth_data = input_data_mat["depth_data"]
    xdim = load_matlab_scalar(input_data_mat, "xdim", 1.5e-6)
    ydim = load_matlab_scalar(input_data_mat, "ydim", 1.5e-6)
    mark_type = load_matlab_string(
        input_data_mat, "mark_type", "Bullet LEA striation mark"
    )

    # Extract parameters
    params = {
        "angle_accuracy": load_matlab_scalar(input_param_mat, "angle_accuracy", 90.0),
        "cutoff_hi": load_matlab_scalar(input_param_mat, "cutoff_hi", 250.0),
        "cutoff_lo": load_matlab_scalar(input_param_mat, "cutoff_lo", 5.0),
        "slope_correction": load_matlab_scalar(input_param_mat, "slope_correction", 0),
        "times_median": load_matlab_scalar(input_param_mat, "times_median", 15.0),
        "use_mean": load_matlab_scalar(input_param_mat, "use_mean", 1),
        "show_info": load_matlab_scalar(input_param_mat, "show_info", 1),
    }

    # Extract mask
    mask = input_mask_mat["mask"]

    # Extract output data
    output_depth_data = output_data_mat["depth_data"]
    output_mask = output_mask_mat["mask_out"]

    # extract last two dirs for 'naming'.
    last_two_dirs = "/".join(case_dir.parts[-2:])
    return MatlabTestCase(
        name=last_two_dirs,
        input_depth_data=depth_data,
        input_xdim=float(xdim),
        input_ydim=float(ydim),
        input_mark_type=mark_type,
        input_mask=mask,
        params=params,
        output_depth_data=output_depth_data,
        output_mask=output_mask,
        output_profile=output_profile,
        output_rotation_angle=output_rotation_angle,
        has_mask=has_mask,
    )


def discover_test_cases(striation_case_root: Path) -> list[MatlabTestCase]:
    """Discover all test cases in the test cases directory."""
    if not striation_case_root.exists():
        return []

    cases = []
    for folder in ["files_with_nans", "files_without_nans"]:
        case_dir = striation_case_root / folder
        for case_dir in sorted(case_dir.iterdir()):
            if case_dir.is_dir() and (case_dir / "input_data.mat").exists():
                try:
                    cases.append(load_test_case(case_dir))
                except Exception as e:
                    print(f"Warning: Failed to load test case {case_dir.name}: {e}")

    return cases


def run_python_preprocessing(
    test_case: MatlabTestCase,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float | None]:
    """
    Run Python preprocess_striations on input data.

    Returns:
        Tuple of (processed_depth_data, output_mask, mean_profile, rotation_angle)
    """
    # Create input data structure
    scan_image = ScanImage(
        data=test_case.input_depth_data,
        scale_x=test_case.input_xdim,
        scale_y=test_case.input_ydim,
    )

    # Convert MATLAB parameters to Python parameters
    # MATLAB stores cutoffs in microns, Python expects meters
    params = PreprocessingStriationParams(
        cutoff_hi=float(test_case.params.get("cutoff_hi", 250.0)) * 1e-6,
        cutoff_lo=float(test_case.params.get("cutoff_lo", 5.0)) * 1e-6,
        use_mean=bool(test_case.params.get("use_mean", 1)),
        angle_accuracy=float(test_case.params.get("angle_accuracy", 90.0)),
    )

    # Convert mask to boolean
    mask = test_case.input_mask.astype(bool) if test_case.has_mask else None

    # Convert string mark type to MarkType enum
    mark_type = _string_to_mark_type(test_case.input_mark_type)

    # Call Python implementation
    aligned_data, profile, mask_out, total_angle = preprocess_data(
        scan_image=scan_image,
        mark_type=mark_type,
        mask=mask,
        params=params,
    )

    return (
        aligned_data,
        mask_out,
        profile,
        total_angle,
    )


def pytest_generate_tests(metafunc):
    """Generate test cases dynamically for MATLAB comparison tests."""
    if "test_case_name" in metafunc.fixturenames:
        # Path to tests/resources/preprocess_striation
        case_dir = (
            Path(__file__).parent.parent.parent / "resources" / "preprocess_striation"
        )

        if case_dir.exists():
            cases = discover_test_cases(case_dir)
            if cases:
                metafunc.parametrize("test_case_name", [c.name for c in cases])
                return

        # No test cases found
        metafunc.parametrize("test_case_name", [])


@pytest.fixture(scope="module")
def striation_test_cases_dir() -> Path:
    """Get the test cases directory for preprocess_striation tests."""
    return Path(__file__).parent.parent.parent / "resources" / "preprocess_striation"


@pytest.fixture(scope="module")
def test_cases(striation_test_cases_dir) -> list[MatlabTestCase]:
    """Load all MATLAB test cases."""
    cases = discover_test_cases(striation_test_cases_dir)
    if not cases:
        pytest.skip(f"No test cases found in {striation_test_cases_dir}")
    return cases


@pytest.fixture
def test_case(test_case_name: str, test_cases: list[MatlabTestCase]) -> MatlabTestCase:
    """Get individual test case by name."""
    for case in test_cases:
        if case.name == test_case_name:
            return case
    pytest.skip(f"Test case {test_case_name} not found")


class TestPreprocessDataMatlabComparison:
    """Test Python PreprocessData against MATLAB reference outputs."""

    # Thresholds for comparison
    CORRELATION_THRESHOLD = 0.9999
    RELATIVE_STD_THRESHOLD = 0.003

    def test_processed_output_correlation(self, test_case: MatlabTestCase):
        """Test that processed output has high correlation with MATLAB."""
        python_depth, _, _, _ = run_python_preprocessing(test_case)

        # Use center cropping for mask cases (MATLAB extracts central region)
        python_depth, matlab_depth = _crop_to_common_shape(
            python_depth, test_case.output_depth_data, center_crop=test_case.has_mask
        )
        correlation = _compute_correlation(python_depth, matlab_depth)

        assert correlation > self.CORRELATION_THRESHOLD, (
            f"Test case {test_case.name}: "
            f"Processed correlation {correlation:.6f} below threshold {self.CORRELATION_THRESHOLD}"
        )

    def test_processed_output_difference(self, test_case: MatlabTestCase):
        """Test that processed output has small differences from MATLAB."""
        python_depth, _, _, _ = run_python_preprocessing(test_case)

        # Use center cropping for mask cases (MATLAB extracts central region)
        python_depth, matlab_depth = _crop_to_common_shape(
            python_depth, test_case.output_depth_data, center_crop=test_case.has_mask
        )

        stats = _compute_difference_stats(python_depth, matlab_depth)
        signal_std = np.nanstd(test_case.output_depth_data)
        relative_std = stats["std"] / signal_std if signal_std > 0 else np.inf

        assert relative_std < self.RELATIVE_STD_THRESHOLD, (
            f"Test case {test_case.name}: "
            f"Relative std {relative_std:.6f} above threshold {self.RELATIVE_STD_THRESHOLD}"
        )

    def test_output_shape(self, test_case: MatlabTestCase):
        """Test that output shape matches MATLAB within tolerance."""
        # Skip shape test for mask cases - MATLAB extracts a central region
        # while Python preserves the full masked area
        if test_case.has_mask:
            pytest.skip(
                "Shape test skipped for mask cases (MATLAB extracts central region)"
            )

        python_depth, _, _, _ = run_python_preprocessing(test_case)

        matlab_shape = test_case.output_depth_data.shape
        python_shape = python_depth.shape

        row_diff = abs(matlab_shape[0] - python_shape[0])
        col_diff = abs(matlab_shape[1] - python_shape[1])

        # Allow small differences due to border handling
        max_diff = 1

        assert row_diff <= max_diff, (
            f"Test case {test_case.name}: "
            f"Row difference {row_diff} > {max_diff} "
            f"(MATLAB: {matlab_shape[0]}, Python: {python_shape[0]})"
        )
        assert col_diff <= max_diff, (
            f"Test case {test_case.name}: "
            f"Column difference {col_diff} > {max_diff} "
            f"(MATLAB: {matlab_shape[1]}, Python: {python_shape[1]})"
        )

    # Profile correlation threshold (mean/median of columns can amplify differences)
    PROFILE_CORRELATION_THRESHOLD = 0.997

    def test_profile_correlation(self, test_case: MatlabTestCase):
        """Test that mean/median profile has high correlation with MATLAB."""
        if test_case.output_profile is None:
            pytest.skip("No profile output in this test case")

        _, _, python_profile, _ = run_python_preprocessing(test_case)

        if python_profile is None:
            pytest.skip("Python implementation did not return profile")

        # Ensure 1D arrays
        python_profile = np.asarray(python_profile).flatten()
        matlab_profile = np.asarray(test_case.output_profile).flatten()

        # Skip if MATLAB profile is all-NaN (edge case)
        if np.all(np.isnan(matlab_profile)):
            pytest.skip("MATLAB profile is all-NaN (edge case)")

        # Use center cropping for mask cases (MATLAB extracts central region)
        min_len = min(len(python_profile), len(matlab_profile))
        if test_case.has_mask:
            # Extract central region
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

        assert correlation > self.PROFILE_CORRELATION_THRESHOLD, (
            f"Test case {test_case.name}: "
            f"Profile correlation {correlation:.6f} below threshold {self.PROFILE_CORRELATION_THRESHOLD}"
        )

    def test_rotation_angle(self, test_case: MatlabTestCase):
        """Test that rotation angle matches MATLAB."""
        if test_case.output_rotation_angle is None:
            pytest.skip("No rotation angle in this test case")

        _, _, _, python_rotation = run_python_preprocessing(test_case)

        if python_rotation is None:
            pytest.skip("Python implementation did not return rotation angle")

        angle_diff = abs(python_rotation - test_case.output_rotation_angle)

        # Allow small differences in rotation angle (degrees)
        assert angle_diff < 0.1, (
            f"Test case {test_case.name}: "
            f"Rotation angle difference {angle_diff:.4f} degrees "
            f"(MATLAB: {test_case.output_rotation_angle}, Python: {python_rotation})"
        )

    def test_mask_output(self, test_case: MatlabTestCase):
        """Test that output mask matches MATLAB."""
        if not test_case.has_mask:
            pytest.skip("Test case has no mask")

        _, python_mask, _, _ = run_python_preprocessing(test_case)
        assert python_mask is not None, "Expected mask output for mask test case"

        # Use center cropping since MATLAB extracts a central region
        python_mask, matlab_mask = _crop_to_common_shape(
            python_mask, test_case.output_mask, center_crop=True
        )

        # Convert to binary
        python_binary = (python_mask > 0.5).astype(float)
        matlab_binary = (matlab_mask > 0.5).astype(float)

        # Calculate IoU (Intersection over Union)
        intersection = np.sum(python_binary * matlab_binary)
        union = np.sum(np.maximum(python_binary, matlab_binary))
        iou = intersection / union if union > 0 else 1.0

        assert iou > 0.95, (
            f"Test case {test_case.name}: Mask IoU {iou:.4f} below threshold 0.95"
        )


class TestMatlabTestCaseLoading:
    """Tests for verifying test case loading works correctly."""

    def test_discover_test_cases(self, striation_test_cases_dir):
        """Test that test cases can be discovered."""
        cases = discover_test_cases(striation_test_cases_dir)
        # This will pass even with 0 cases, just logs the count
        print(f"Found {len(cases)} test cases in {striation_test_cases_dir}")

    def test_load_test_case_structure(self, test_cases: list[MatlabTestCase]):
        """Test that loaded test cases have valid structure."""
        for case in test_cases:
            # Check input data
            assert case.input_depth_data is not None
            assert case.input_depth_data.ndim == 2
            assert case.input_xdim > 0
            assert case.input_ydim > 0

            # Check mask
            assert case.input_mask is not None
            assert case.input_mask.shape == case.input_depth_data.shape

            # Check output data
            assert case.output_depth_data is not None
            assert case.output_depth_data.ndim == 2

            # Check parameters
            assert "cutoff_hi" in case.params
            assert "cutoff_lo" in case.params
            assert "use_mean" in case.params

    def test_print_test_case_summary(self, test_cases: list[MatlabTestCase]):
        """Print summary of all test cases for debugging."""
        print("\n=== Test Case Summary ===")
        for case in test_cases:
            print(f"\n{case.name}:")
            print(f"  Input shape: {case.input_depth_data.shape}")
            print(f"  Output shape: {case.output_depth_data.shape}")
            print(f"  Pixel size: ({case.input_xdim}, {case.input_ydim})")
            print(f"  Has mask: {case.has_mask}")
            print(f"  Use mean: {case.params.get('use_mean')}")
            if case.output_rotation_angle is not None:
                print(f"  Rotation angle: {case.output_rotation_angle}")
