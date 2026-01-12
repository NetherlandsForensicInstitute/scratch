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


# # TODO: Update these imports to match your actual module structure
# from conversion.preprocess_striations.preprocess_data import preprocess_data
# from conversion.preprocess_striations.parameters import PreprocessingStriationParams
# from container_models.scan_image import ScanImage
# from conversion.data_formats import Mark, MarkType, CropType


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
        "shape_noise_removal": load_matlab_scalar(
            input_param_mat, "shape_noise_removal", 1
        ),
        "show_info": load_matlab_scalar(input_param_mat, "show_info", 1),
    }

    # Extract mask
    mask = input_mask_mat["mask"]

    # Extract output data
    output_depth_data = output_data_mat["depth_data"]
    output_mask = output_mask_mat["mask_out"]

    return MatlabTestCase(
        name=case_dir.name,
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


def discover_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Discover all test cases in the test cases directory."""
    if not test_cases_dir.exists():
        return []

    cases = []
    for case_dir in sorted(test_cases_dir.iterdir()):
        if case_dir.is_dir() and (case_dir / "input_data.mat").exists():
            try:
                cases.append(load_test_case(case_dir))
            except Exception as e:
                print(f"Warning: Failed to load test case {case_dir.name}: {e}")

    return cases


# TODO: Uncomment and update this function once your Python implementation is ready
# def matlab_params_to_python_params(
#     matlab_params: dict,
# ) -> PreprocessingStriationParams:
#     """
#     Convert MATLAB parameters to Python PreprocessingStriationParams.
#     """
#     return PreprocessingStriationParams(
#         angle_accuracy=float(matlab_params.get("angle_accuracy", 90.0)),
#         cutoff_hi=float(matlab_params.get("cutoff_hi", 250.0)),
#         cutoff_lo=float(matlab_params.get("cutoff_lo", 5.0)),
#         slope_correction=bool(matlab_params.get("slope_correction", 0)),
#         times_median=float(matlab_params.get("times_median", 15.0)),
#         use_mean=bool(matlab_params.get("use_mean", 1)),
#         shape_noise_removal=bool(matlab_params.get("shape_noise_removal", 1)),
#     )


# TODO: Uncomment and update this function once your Python implementation is ready
# def run_python_preprocessing(
#     test_case: MatlabTestCase,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float | None]:
#     """
#     Run Python preprocess_striations on input data.
#
#     Returns:
#         Tuple of (processed_depth_data, output_mask, mean_profile, rotation_angle)
#     """
#     params = matlab_params_to_python_params(test_case.params)
#
#     # Create input data structure matching your Python implementation
#     # This will depend on your actual data model
#     scan_image = ScanImage(
#         data=test_case.input_depth_data,
#         scale_x=test_case.input_xdim,
#         scale_y=test_case.input_ydim,
#     )
#
#     mark = Mark(
#         scan_image=scan_image,
#         mark_type=MarkType.BULLET_LEA_STRIATION,
#         crop_type=CropType.RECTANGLE,
#     )
#
#     # Call your Python implementation
#     result = preprocess_data(
#         data=mark,
#         param=params,
#         mask=test_case.input_mask,
#         data_type='bullet',
#     )
#
#     return (
#         result.depth_data,
#         result.mask,
#         result.mean_profile,
#         result.rotation_angle,
#     )


def pytest_generate_tests(metafunc):
    """Generate test cases dynamically for MATLAB comparison tests."""
    if "test_case_name" in metafunc.fixturenames:
        # Path from tests/conversion/test_file.py to tests/resources/baseline_images/preprocess_striation
        test_cases_dir = (
            Path(__file__).parent.parent
            / "resources"
            / "baseline_images"
            / "preprocess_striation"
        )

        if test_cases_dir.exists():
            cases = discover_test_cases(test_cases_dir)
            if cases:
                metafunc.parametrize("test_case_name", [c.name for c in cases])
                return

        # No test cases found
        metafunc.parametrize("test_case_name", [])


@pytest.fixture(scope="module")
def test_cases_dir(baseline_images_dir) -> Path:
    """Get the test cases directory."""
    return baseline_images_dir / "preprocess_striation"


@pytest.fixture(scope="module")
def test_cases(test_cases_dir) -> list[MatlabTestCase]:
    """Load all MATLAB test cases."""
    cases = discover_test_cases(test_cases_dir)
    if not cases:
        pytest.skip(f"No test cases found in {test_cases_dir}")
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
    CORRELATION_THRESHOLD = 0.999
    RELATIVE_STD_THRESHOLD = 0.01

    # Looser thresholds for cases with masks (edge effects)
    CORRELATION_THRESHOLD_MASKED = 0.99
    RELATIVE_STD_THRESHOLD_MASKED = 0.05

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get correlation and relative std thresholds based on test case."""
        if test_case.has_mask:
            return (
                self.CORRELATION_THRESHOLD_MASKED,
                self.RELATIVE_STD_THRESHOLD_MASKED,
            )
        return self.CORRELATION_THRESHOLD, self.RELATIVE_STD_THRESHOLD

    @pytest.mark.skip(reason="Python implementation not yet available")
    def test_processed_output_correlation(self, test_case: MatlabTestCase):
        """Test that processed output has high correlation with MATLAB."""
        python_depth, _, _, _ = run_python_preprocessing(test_case)

        python_depth, matlab_depth = _crop_to_common_shape(
            python_depth, test_case.output_depth_data
        )
        correlation = _compute_correlation(python_depth, matlab_depth)
        corr_threshold, _ = self._get_thresholds(test_case)

        assert correlation > corr_threshold, (
            f"Test case {test_case.name}: "
            f"Processed correlation {correlation:.6f} below threshold {corr_threshold}"
            f"{' (masked)' if test_case.has_mask else ''}"
        )

    @pytest.mark.skip(reason="Python implementation not yet available")
    def test_processed_output_difference(self, test_case: MatlabTestCase):
        """Test that processed output has small differences from MATLAB."""
        python_depth, _, _, _ = run_python_preprocessing(test_case)

        python_depth, matlab_depth = _crop_to_common_shape(
            python_depth, test_case.output_depth_data
        )
        stats = _compute_difference_stats(python_depth, matlab_depth)
        signal_std = np.nanstd(test_case.output_depth_data)
        relative_std = stats["std"] / signal_std if signal_std > 0 else np.inf

        _, std_threshold = self._get_thresholds(test_case)

        assert relative_std < std_threshold, (
            f"Test case {test_case.name}: "
            f"Relative std {relative_std:.6f} above threshold {std_threshold}"
            f"{' (masked)' if test_case.has_mask else ''}"
        )

    @pytest.mark.skip(reason="Python implementation not yet available")
    def test_output_shape(self, test_case: MatlabTestCase):
        """Test that output shape matches MATLAB within tolerance."""
        python_depth, _, _, _ = run_python_preprocessing(test_case)

        matlab_shape = test_case.output_depth_data.shape
        python_shape = python_depth.shape

        row_diff = abs(matlab_shape[0] - python_shape[0])
        col_diff = abs(matlab_shape[1] - python_shape[1])

        # Allow small differences due to border handling
        max_diff = 2 if test_case.has_mask else 1

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

    @pytest.mark.skip(reason="Python implementation not yet available")
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

        # Crop to common length
        min_len = min(len(python_profile), len(matlab_profile))
        python_profile = python_profile[:min_len]
        matlab_profile = matlab_profile[:min_len]

        correlation = _compute_correlation(
            python_profile.reshape(-1, 1), matlab_profile.reshape(-1, 1)
        )
        corr_threshold, _ = self._get_thresholds(test_case)

        assert correlation > corr_threshold, (
            f"Test case {test_case.name}: "
            f"Profile correlation {correlation:.6f} below threshold {corr_threshold}"
        )

    @pytest.mark.skip(reason="Python implementation not yet available")
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

    @pytest.mark.skip(reason="Python implementation not yet available")
    def test_mask_output(self, test_case: MatlabTestCase):
        """Test that output mask matches MATLAB."""
        if not test_case.has_mask:
            pytest.skip("Test case has no mask")

        _, python_mask, _, _ = run_python_preprocessing(test_case)

        python_mask, matlab_mask = _crop_to_common_shape(
            python_mask, test_case.output_mask
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

    def test_discover_test_cases(self, test_cases_dir):
        """Test that test cases can be discovered."""
        cases = discover_test_cases(test_cases_dir)
        # This will pass even with 0 cases, just logs the count
        print(f"Found {len(cases)} test cases in {test_cases_dir}")

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
            assert "shape_noise_removal" in case.params

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
            print(f"  Shape/noise removal: {case.params.get('shape_noise_removal')}")
            if case.output_rotation_angle is not None:
                print(f"  Rotation angle: {case.output_rotation_angle}")
