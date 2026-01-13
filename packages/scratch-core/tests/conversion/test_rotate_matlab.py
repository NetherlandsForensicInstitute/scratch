"""
This file contains tests to explicitly compare the Python rotate_crop_image output
with the original MATLAB RotateCropImage output.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from container_models.scan_image import ScanImage
from conversion.data_formats import CropType, CropInfo

# Individual functions used for debugging/step-by-step comparison
from conversion.rotate import (
    get_rotation_angle,
    dilate_and_crop_image_and_mask,
    rotate_and_crop_scan_image,
)
from tests.conversion.helper_function import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)


from conversion.rotate import rotate_crop_image_full_flow


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB RotateCropImage test case."""

    name: str
    # Input data
    input_depth_data: np.ndarray
    input_xdim: float
    input_ydim: float
    input_mask: np.ndarray
    # Crop info
    rotation_angle_input: float
    crop_type: str
    crop_info_corners: np.ndarray | None
    crop_info_foreground: bool
    # Parameters
    times_median: float
    # Output data
    output_depth_data: np.ndarray
    output_mask: np.ndarray
    # Metadata
    has_holes: bool


def load_matlab_scalar(mat_dict: dict, key: str, default: Any = None) -> Any:
    """Extract a scalar value from loaded .mat dictionary."""
    if key not in mat_dict:
        return default
    val = mat_dict[key]
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
        if val.dtype.kind in ("U", "S"):
            return "".join(val.flatten())
        elif val.size == 1:
            return str(val.flat[0])
    return str(val) if val is not None else default


def load_crop_info(case_dir: Path) -> tuple[str, np.ndarray | None, bool]:
    """
    Load crop_info from MATLAB cell array.

    Returns:
        Tuple of (crop_type, corners, is_foreground)
    """
    crop_info_mat = loadmat(case_dir / "input_crop_info.mat")
    crop_info = crop_info_mat["crop_info"]

    # MATLAB cell array structure: {crop_type, struct with corner, foreground_flag}
    # crop_info is a 1x3 cell array
    if crop_info.size == 0:
        return "rectangle", None, True

    # Extract crop type
    crop_type_cell = crop_info[0, 0]
    if isinstance(crop_type_cell, np.ndarray):
        crop_type = "".join(crop_type_cell.flatten())
    else:
        crop_type = str(crop_type_cell)

    # Extract corners from struct
    corners = None
    struct_cell = crop_info[0, 1]
    if isinstance(struct_cell, np.ndarray) and struct_cell.dtype.names:
        if "corner" in struct_cell.dtype.names:
            corners = struct_cell["corner"].flat[0]
            if isinstance(corners, np.ndarray):
                corners = corners.astype(float)

    # Extract foreground flag
    fg_cell = crop_info[0, 2]
    if isinstance(fg_cell, np.ndarray):
        is_foreground = bool(fg_cell.flat[0])
    else:
        is_foreground = bool(fg_cell)

    return crop_type, corners, is_foreground


def load_test_case(case_dir: Path) -> MatlabTestCase:
    """Load a single test case from a directory."""
    # Load input data
    input_data_mat = loadmat(case_dir / "input_data.mat")
    input_mask_mat = loadmat(case_dir / "input_mask.mat")
    input_param_mat = loadmat(case_dir / "input_param.mat")
    input_rotation_mat = loadmat(case_dir / "input_rotation_angle.mat")
    metadata_mat = loadmat(case_dir / "metadata.mat")

    # Load output data
    output_data_mat = loadmat(case_dir / "output_data.mat")
    output_mask_mat = loadmat(case_dir / "output_mask.mat")

    # Extract input data
    depth_data = input_data_mat["depth_data"]
    xdim = load_matlab_scalar(input_data_mat, "xdim", 3.5e-6)
    ydim = load_matlab_scalar(input_data_mat, "ydim", 3.5e-6)
    mask = input_mask_mat["mask"].astype(bool)

    # Extract rotation angle
    rotation_angle_input = load_matlab_scalar(input_rotation_mat, "rotation_angle", 0.0)

    # Extract crop info
    crop_type, corners, is_foreground = load_crop_info(case_dir)

    # Extract parameters
    times_median = load_matlab_scalar(input_param_mat, "times_median", 15.0)

    # Extract metadata
    has_holes = bool(load_matlab_scalar(metadata_mat, "has_holes", 0))

    # Extract output data
    output_depth_data = output_data_mat["depth_data"]
    output_mask = output_mask_mat["mask_out"].astype(bool)

    return MatlabTestCase(
        name=case_dir.name,
        input_depth_data=depth_data,
        input_xdim=float(xdim),
        input_ydim=float(ydim),
        input_mask=mask,
        rotation_angle_input=float(rotation_angle_input),
        crop_type=crop_type,
        crop_info_corners=corners,
        crop_info_foreground=is_foreground,
        times_median=float(times_median),
        output_depth_data=output_depth_data,
        output_mask=output_mask,
        has_holes=has_holes,
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


def matlab_crop_info_to_python(test_case: MatlabTestCase) -> list[CropInfo] | None:
    """
    Convert MATLAB crop_info to Python CropInfo list.
    """
    if test_case.crop_info_corners is None:
        return None

    crop_type = (
        CropType.RECTANGLE
        if test_case.crop_type.lower() == "rectangle"
        else CropType.POLYGON
    )

    crop_info = CropInfo(
        crop_type=crop_type,
        data={"corner": test_case.crop_info_corners},
        is_foreground=test_case.crop_info_foreground,
    )

    return [crop_info]


def run_python_preprocessing(
    test_case: MatlabTestCase,
    debug: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Python rotate_crop_image on input data.

    Returns:
        Tuple of (processed_depth_data, output_mask)
    """
    # Create ScanImage
    scan_image = ScanImage(
        data=test_case.input_depth_data.copy(),
        scale_x=test_case.input_xdim,
        scale_y=test_case.input_ydim,
    )

    mask = test_case.input_mask.copy()

    # Convert crop_info
    crop_info = matlab_crop_info_to_python(test_case)

    if debug:
        print(f"\n=== Debug: {test_case.name} ===")
        print(f"Input shape: {test_case.input_depth_data.shape}")
        print(f"Rotation angle input: {test_case.rotation_angle_input}")
        print(f"Crop type: {test_case.crop_type}")
        print(f"Has holes: {test_case.has_holes}")
        print(f"MATLAB output shape: {test_case.output_depth_data.shape}")

    data_out, mask_out = rotate_crop_image_full_flow(
        scan_image=scan_image,
        mask=mask,
        rotation_angle=test_case.rotation_angle_input,
        crop_info=crop_info,
        times_median=test_case.times_median,
    )
    return data_out.data, mask_out

    # --- Current step-by-step implementation for debugging ---

    # Step 1: Calculate rotation angle (if not provided)
    rotation_angle = get_rotation_angle(
        rotation_angle=test_case.rotation_angle_input,
        crop_info=crop_info,
    )

    if debug:
        print(f"Calculated rotation angle: {rotation_angle}")

    # Step 2: Dilate mask and crop to bounding box
    scan_image_cropped, mask_cropped = dilate_and_crop_image_and_mask(
        scan_image=scan_image,
        mask=mask,
        rotation_angle=rotation_angle,
    )

    if debug:
        print(f"After crop shape: {scan_image_cropped.data.shape}")

    # Step 3: TODO - Remove holes and stitches
    # This corresponds to RemoveHolesAndStitchesNew in MATLAB
    # from conversion.remove_holes import remove_holes_and_stitches
    # scan_image_no_holes = remove_holes_and_stitches(
    #     scan_image_cropped, mask_cropped, interpolate=test_case.interpolate_data
    # )
    scan_image_no_holes = scan_image_cropped

    # Step 4: Set NaN outside mask
    data_masked = scan_image_no_holes.data.copy()
    if mask_cropped is not None:
        data_masked[mask_cropped == 0] = np.nan

    # Step 5: Rotate if needed
    if rotation_angle != 0:
        scan_image_masked = ScanImage(
            data=data_masked,
            scale_x=scan_image_no_holes.scale_x,
            scale_y=scan_image_no_holes.scale_y,
        )
        scan_image_rotated, mask_rotated = rotate_and_crop_scan_image(
            scan_image_masked, mask_cropped, rotation_angle
        )
        data_out = scan_image_rotated.data

        # TODO: Rotate and crop mask similarly
        mask_out = mask_cropped  # Placeholder
    else:
        data_out = data_masked
        mask_out = mask_cropped

    if debug:
        print(f"Python output shape: {data_out.shape}")
        print(
            f"Python output range: [{np.nanmin(data_out):.6e}, {np.nanmax(data_out):.6e}]"
        )
        print(
            f"MATLAB output range: [{np.nanmin(test_case.output_depth_data):.6e}, {np.nanmax(test_case.output_depth_data):.6e}]"
        )

    return data_out, mask_out


def pytest_generate_tests(metafunc):
    """Generate test cases dynamically for MATLAB comparison tests."""
    if "test_case_name" in metafunc.fixturenames:
        test_cases_dir = (
            Path(__file__).parent.parent
            / "resources"
            / "baseline_images"
            / "rotate_crop_image"
        )

        if test_cases_dir.exists():
            cases = discover_test_cases(test_cases_dir)
            if cases:
                metafunc.parametrize("test_case_name", [c.name for c in cases])
                return

        metafunc.parametrize("test_case_name", [])


@pytest.fixture(scope="module")
def test_cases_dir(baseline_images_dir) -> Path:
    """Get the test cases directory."""
    return baseline_images_dir / "rotate_crop_image"


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


class TestRotateCropImageMatlabComparison:
    """Test Python rotate_crop_image against MATLAB reference outputs."""

    # Thresholds for comparison (no rotation)
    CORRELATION_THRESHOLD = 0.999
    RELATIVE_STD_THRESHOLD = 0.01

    # Looser thresholds for cases with rotation (interpolation differences)
    CORRELATION_THRESHOLD_ROTATED = 0.95
    RELATIVE_STD_THRESHOLD_ROTATED = 0.25

    # Looser thresholds for cases with holes
    CORRELATION_THRESHOLD_HOLES = 0.95
    RELATIVE_STD_THRESHOLD_HOLES = 0.20

    # Combined rotation + holes
    CORRELATION_THRESHOLD_ROTATED_HOLES = 0.90
    RELATIVE_STD_THRESHOLD_ROTATED_HOLES = 0.30

    @staticmethod
    def _involves_rotation(test_case: MatlabTestCase) -> bool:
        """Check if test case involves rotation."""
        # Rotation is involved if input angle != 0 or if corners suggest rotation
        if test_case.rotation_angle_input != 0:
            return True
        # Check if "rotation" is in the name (from our test naming)
        if "rotation" in test_case.name.lower():
            return True
        return False

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get correlation and relative std thresholds based on test case."""
        is_rotated = self._involves_rotation(test_case)
        has_holes = test_case.has_holes

        if is_rotated and has_holes:
            return (
                self.CORRELATION_THRESHOLD_ROTATED_HOLES,
                self.RELATIVE_STD_THRESHOLD_ROTATED_HOLES,
            )
        if has_holes:
            return (
                self.CORRELATION_THRESHOLD_HOLES,
                self.RELATIVE_STD_THRESHOLD_HOLES,
            )
        if is_rotated:
            return (
                self.CORRELATION_THRESHOLD_ROTATED,
                self.RELATIVE_STD_THRESHOLD_ROTATED,
            )
        return self.CORRELATION_THRESHOLD, self.RELATIVE_STD_THRESHOLD

    def test_processed_output_correlation(self, test_case: MatlabTestCase):
        """Test that processed output has high correlation with MATLAB."""
        python_data, _ = run_python_preprocessing(test_case)

        python_data, matlab_data = _crop_to_common_shape(
            python_data, test_case.output_depth_data
        )

        # Handle NaN values
        valid_mask = ~np.isnan(python_data) & ~np.isnan(matlab_data)
        if np.sum(valid_mask) < 10:
            pytest.skip("Too few valid pixels for comparison")

        correlation = _compute_correlation(python_data, matlab_data)
        corr_threshold, _ = self._get_thresholds(test_case)

        assert correlation > corr_threshold, (
            f"Test case {test_case.name}: "
            f"Correlation {correlation:.6f} below threshold {corr_threshold}"
            f"{' (rotated)' if self._involves_rotation(test_case) else ''}"
            f"{' (holes)' if test_case.has_holes else ''}"
        )

    def test_processed_output_difference(self, test_case: MatlabTestCase):
        """Test that processed output has small differences from MATLAB."""
        python_data, _ = run_python_preprocessing(test_case)

        python_data, matlab_data = _crop_to_common_shape(
            python_data, test_case.output_depth_data
        )

        stats = _compute_difference_stats(python_data, matlab_data)
        signal_std = np.nanstd(test_case.output_depth_data)
        relative_std = stats["std"] / signal_std if signal_std > 0 else np.inf

        _, std_threshold = self._get_thresholds(test_case)

        assert relative_std < std_threshold, (
            f"Test case {test_case.name}: "
            f"Relative std {relative_std:.6f} above threshold {std_threshold}"
            f"{' (rotated)' if self._involves_rotation(test_case) else ''}"
            f"{' (holes)' if test_case.has_holes else ''}"
        )

    def test_output_shape(self, test_case: MatlabTestCase):
        """Test that output shape matches MATLAB within tolerance."""
        python_data, _ = run_python_preprocessing(test_case)

        matlab_shape = test_case.output_depth_data.shape
        python_shape = python_data.shape

        row_diff = abs(matlab_shape[0] - python_shape[0])
        col_diff = abs(matlab_shape[1] - python_shape[1])

        # Allow larger differences for rotated cases
        max_diff = 5 if self._involves_rotation(test_case) else 2

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

    def test_rotation_angle_calculation(self, test_case: MatlabTestCase):
        """Test that rotation angle is calculated correctly from corners."""
        if test_case.rotation_angle_input != 0:
            pytest.skip("Rotation angle was pre-specified")

        crop_info = matlab_crop_info_to_python(test_case)

        calculated_angle = get_rotation_angle(
            rotation_angle=0.0,
            crop_info=crop_info,
        )

        # For test cases with known rotation, check against expected
        # Extract expected angle from test name if present
        if "rotation" in test_case.name.lower():
            # Test name format: XX_rotation_YYdeg
            import re

            match = re.findall(r"rotation_?(neg)?(\d+)deg", test_case.name.lower())
            if match:
                expected = float(match[0][1])
                if match[0][0]:  # neg prefix
                    expected = -expected

                assert abs(calculated_angle - expected) < 1.0, (
                    f"Test case {test_case.name}: "
                    f"Calculated angle {calculated_angle:.2f} != expected {expected}"
                )


class TestMatlabTestCaseLoading:
    """Tests for verifying test case loading works correctly."""

    def test_discover_test_cases(self, test_cases_dir):
        """Test that test cases can be discovered."""
        cases = discover_test_cases(test_cases_dir)
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

            # Check crop type
            assert case.crop_type.lower() in ("rectangle", "polygon")

    def test_print_test_case_summary(self, test_cases: list[MatlabTestCase]):
        """Print summary of all test cases for debugging."""
        print("\n=== Test Case Summary ===")
        for case in test_cases:
            print(f"\n{case.name}:")
            print(f"  Input shape: {case.input_depth_data.shape}")
            print(f"  Output shape: {case.output_depth_data.shape}")
            print(f"  Rotation angle input: {case.rotation_angle_input}")
            print(f"  Crop type: {case.crop_type}")
            print(f"  Has holes: {case.has_holes}")
