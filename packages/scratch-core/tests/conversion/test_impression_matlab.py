"""
This file contains tests to explicitly compare the Python output with the original Matlab output. Upon completion of
the translation and incorporation of the , it will be redundant and can be removed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType, CropType
from conversion.preprocess_impression.impression import preprocess_impression_mark
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from .helper_function import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB test case."""

    name: str
    input_map: np.ndarray
    input_pixel_spacing: tuple[float, float]
    params: dict[str, Any]
    output_processed: np.ndarray
    output_leveled: np.ndarray | None
    b_interpol: bool


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


def load_filter_params(input_par_mat: dict) -> list[dict]:
    """
    Load filter parameters from MATLAB struct array.

    MATLAB Filter struct has fields:
    - type: 'Gaussian'
    - nOrder: 0 or 2
    - bRobust: 0 or 1
    - vCutoffLength: [x, y]
    - bHighPass: 0 or 1
    - domain: 'rectangle'
    - vSize: []
    """
    if "Filter" not in input_par_mat:
        return []

    filter_data = input_par_mat["Filter"]

    # Handle empty filter
    if filter_data.size == 0:
        return []

    filters = []

    # MATLAB struct arrays are loaded as numpy arrays with structured dtype
    # Shape could be (1, N) or (N,) depending on scipy version
    if filter_data.ndim == 2:
        n_filters = filter_data.shape[1]
        for i in range(n_filters):
            f = filter_data[0, i]
            filters.append(_extract_filter_struct(f))
    elif filter_data.ndim == 1:
        for i in range(len(filter_data)):
            f = filter_data[i]
            filters.append(_extract_filter_struct(f))
    else:
        # Single filter case
        filters.append(_extract_filter_struct(filter_data))

    return filters


def _extract_filter_struct(f) -> dict:
    """Extract filter parameters from a single MATLAB struct."""

    def get_field(struct, name, default):
        """Safely get a field from MATLAB struct."""
        try:
            if (
                hasattr(struct, "dtype")
                and struct.dtype.names
                and name in struct.dtype.names
            ):
                val = struct[name]
                # Unwrap nested arrays
                while isinstance(val, np.ndarray) and val.ndim > 0 and val.size == 1:
                    val = val.flat[0]
                return val
            elif isinstance(struct, dict) and name in struct:
                return struct[name]
        except (KeyError, IndexError, TypeError):
            pass
        return default

    # Extract cutoff length
    v_cutoff = get_field(f, "vCutoffLength", [0, 0])
    if isinstance(v_cutoff, np.ndarray):
        v_cutoff = v_cutoff.flatten().tolist()

    # Extract nOrder
    n_order = get_field(f, "nOrder", 0)
    if isinstance(n_order, np.ndarray):
        n_order = int(n_order.flat[0])

    # Extract bHighPass
    b_highpass = get_field(f, "bHighPass", 0)
    if isinstance(b_highpass, np.ndarray):
        b_highpass = bool(b_highpass.flat[0])

    return {
        "type": "Gaussian",
        "nOrder": int(n_order),
        "vCutoffLength": v_cutoff,
        "bHighPass": bool(b_highpass),
    }


def load_test_case(case_dir: Path) -> MatlabTestCase:
    """Load a single test case from a directory."""
    # Load input
    input_map_mat = loadmat(case_dir / "input_map.mat")
    input_par_mat = loadmat(case_dir / "input_par.mat")

    # Load outputs
    output_map_mat = loadmat(case_dir / "output_map.mat")
    output_full_mat = loadmat(case_dir / "output_full.mat")

    # Load leveled output if it exists
    output_leveled = None
    if (case_dir / "output_mapNF.mat").exists():
        output_leveled_mat = loadmat(case_dir / "output_mapNF.mat")
        output_leveled = output_leveled_mat.get("mapNF")

    # Extract pixel spacing
    v_pix_sep = load_matlab_scalar(input_map_mat, "vPixSep", [1, 1])
    if isinstance(v_pix_sep, (list, np.ndarray)):
        pixel_spacing = (float(v_pix_sep[0]), float(v_pix_sep[1]))
    else:
        pixel_spacing = (float(v_pix_sep), float(v_pix_sep))

    # Extract parameters
    params = {
        "bMapCenterCircle": load_matlab_scalar(input_par_mat, "bMapCenterCircle", 0),
        "bAdjustPixelSpacing": load_matlab_scalar(
            input_par_mat, "bAdjustPixelSpacing", 0
        ),
        "viParLevel": load_matlab_scalar(input_par_mat, "viParLevel", []),
        "vPixSep": load_matlab_scalar(input_par_mat, "vPixSep", [np.nan, np.nan]),
        "intMeth": load_matlab_scalar(input_par_mat, "intMeth", "linear"),
    }

    # Load filter parameters
    params["filters"] = load_filter_params(input_par_mat)

    # Extract bInterpol from output
    b_interpol = bool(load_matlab_scalar(output_full_mat, "bInterpol", 0))

    return MatlabTestCase(
        name=case_dir.name,
        input_map=input_map_mat["map"],
        input_pixel_spacing=pixel_spacing,
        params=params,
        output_processed=output_map_mat["map"],
        output_leveled=output_leveled,
        b_interpol=b_interpol,
    )


def discover_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Discover all test cases in the test cases directory."""
    if not test_cases_dir.exists():
        return []

    cases = []
    for case_dir in sorted(test_cases_dir.iterdir()):
        if case_dir.is_dir() and (case_dir / "input_map.mat").exists():
            try:
                cases.append(load_test_case(case_dir))
            except Exception as e:
                print(f"Warning: Failed to load test case {case_dir.name}: {e}")

    return cases


def matlab_params_to_python_params(
    matlab_params: dict,
) -> PreprocessingImpressionParams:
    """
    Convert MATLAB parameters to Python PreprocessingImpressionParams.
    """
    # Convert viParLevel to surface terms
    vi_par_level = matlab_params.get("viParLevel", [])

    # Handle numpy scalars and arrays - convert to Python list
    if isinstance(vi_par_level, np.ndarray):
        vi_par_level = vi_par_level.flatten().tolist()
    elif isinstance(vi_par_level, (int, float, np.integer, np.floating)):
        vi_par_level = [int(vi_par_level)]
    elif not isinstance(vi_par_level, list):
        try:
            vi_par_level = list(vi_par_level)
        except TypeError:
            vi_par_level = [vi_par_level]

    level_offset = 1 in vi_par_level
    level_tilt = 2 in vi_par_level and 3 in vi_par_level
    level_2nd = 4 in vi_par_level and 5 in vi_par_level and 6 in vi_par_level

    # Convert vPixSep to pixel_size
    v_pix_sep = matlab_params.get("vPixSep", [np.nan, np.nan])
    if isinstance(v_pix_sep, (list, np.ndarray)) and len(v_pix_sep) >= 2:
        if np.isnan(v_pix_sep[0]) or np.isnan(v_pix_sep[1]):
            pixel_size = None
        else:
            pixel_size = (float(v_pix_sep[0]), float(v_pix_sep[1]))
    else:
        pixel_size = None

    # Convert filter parameters
    highpass_cutoff = None
    lowpass_cutoff = None
    regression_order_high = 2
    regression_order_low = 0

    filters = matlab_params.get("filters", [])
    for f in filters:
        cutoff = f.get("vCutoffLength", [0, 0])
        # Use first cutoff value (assuming symmetric or taking x-direction)
        if isinstance(cutoff, (list, np.ndarray)) and len(cutoff) >= 1:
            cutoff_val = float(cutoff[0])
        else:
            cutoff_val = float(cutoff)  # type: ignore[arg-type]

        if f.get("bHighPass", False):
            highpass_cutoff = cutoff_val
            regression_order_high = int(f.get("nOrder", 2))
        else:
            lowpass_cutoff = cutoff_val
            regression_order_low = int(f.get("nOrder", 0))

    return PreprocessingImpressionParams(
        adjust_pixel_spacing=bool(matlab_params.get("bAdjustPixelSpacing", 0)),
        level_offset=level_offset,
        level_tilt=level_tilt,
        level_2nd=level_2nd,
        pixel_size=pixel_size,
        highpass_cutoff=highpass_cutoff,
        lowpass_cutoff=lowpass_cutoff,
        highpass_regression_order=regression_order_high,
        lowpass_regression_order=regression_order_low,
    )


def run_python_preprocessing(
    input_map: np.ndarray,
    pixel_spacing: tuple[float, float],
    matlab_params: dict,
    use_circle_center: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run Python preprocessing on input data.

    Returns:
        Tuple of (processed_output, leveled_output)
    """
    params = matlab_params_to_python_params(matlab_params)

    scan_image = ScanImage(
        data=input_map,
        scale_x=pixel_spacing[0],
        scale_y=pixel_spacing[1],
    )

    mark_type = (
        MarkType.BREECH_FACE_IMPRESSION
        if use_circle_center
        else MarkType.FIRING_PIN_IMPRESSION
    )
    mark = Mark(
        scan_image=scan_image, mark_type=mark_type, crop_type=CropType.RECTANGLE
    )

    processed, leveled = preprocess_impression_mark(mark, params)

    return processed.scan_image.data, leveled.scan_image.data


def pytest_generate_tests(metafunc):
    """Generate test cases dynamically for MATLAB comparison tests."""
    if "test_case_name" in metafunc.fixturenames:
        # At collection time, fixtures aren't available yet.
        # Path from tests/conversion/test_file.py to tests/resources/baseline_images/preprocess_impression
        test_cases_dir = (
            Path(__file__).parent.parent
            / "resources"
            / "baseline_images"
            / "preprocess_impression"
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
    return baseline_images_dir / "preprocess_impression"


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


class TestMatlabComparison:
    """Test Python preprocessing against MATLAB reference outputs."""

    # Thresholds for tests without resampling (should be near-perfect)
    CORRELATION_THRESHOLD_EXACT = 0.9999
    RELATIVE_STD_THRESHOLD_EXACT = 0.001

    # Thresholds for tests with resampling (interpolation differences expected)
    CORRELATION_THRESHOLD_RESAMPLING = 0.97
    RELATIVE_STD_THRESHOLD_RESAMPLING = 0.2

    @staticmethod
    def _involves_resampling(test_case: MatlabTestCase) -> bool:
        """Check if test case involves resampling."""
        v_pix_sep = test_case.params.get("vPixSep", [np.nan, np.nan])
        if isinstance(v_pix_sep, (list, np.ndarray)) and len(v_pix_sep) >= 2:
            # Resampling if vPixSep is specified and not NaN
            if not (np.isnan(v_pix_sep[0]) or np.isnan(v_pix_sep[1])):
                # Check if it differs from input spacing
                input_spacing = test_case.input_pixel_spacing
                if (
                    abs(v_pix_sep[0] - input_spacing[0]) > 1e-9
                    or abs(v_pix_sep[1] - input_spacing[1]) > 1e-9
                ):
                    return True
        # Also check bInterpol flag
        return test_case.b_interpol

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get correlation and relative std thresholds based on test case."""
        if self._involves_resampling(test_case):
            return (
                self.CORRELATION_THRESHOLD_RESAMPLING,
                self.RELATIVE_STD_THRESHOLD_RESAMPLING,
            )
        return self.CORRELATION_THRESHOLD_EXACT, self.RELATIVE_STD_THRESHOLD_EXACT

    def test_processed_output_correlation(self, test_case: MatlabTestCase):
        """Test that processed output has high correlation with MATLAB."""
        python_processed, _ = run_python_preprocessing(
            test_case.input_map,
            test_case.input_pixel_spacing,
            test_case.params,
            use_circle_center=bool(test_case.params.get("bMapCenterCircle", 0)),
        )

        python_processed, matlab_processed = _crop_to_common_shape(
            python_processed, test_case.output_processed
        )
        correlation = _compute_correlation(python_processed, matlab_processed)
        corr_threshold, _ = self._get_thresholds(test_case)

        assert correlation > corr_threshold, (
            f"Test case {test_case.name}: "
            f"Processed correlation {correlation:.6f} below threshold {corr_threshold}"
            f"{' (resampling involved)' if self._involves_resampling(test_case) else ''}"
        )

    def test_processed_output_difference(self, test_case: MatlabTestCase):
        """Test that processed output has small differences from MATLAB."""
        python_processed, _ = run_python_preprocessing(
            test_case.input_map,
            test_case.input_pixel_spacing,
            test_case.params,
            use_circle_center=bool(test_case.params.get("bMapCenterCircle", 0)),
        )

        python_processed, matlab_processed = _crop_to_common_shape(
            python_processed, test_case.output_processed
        )
        stats = _compute_difference_stats(python_processed, matlab_processed)
        signal_std = np.nanstd(test_case.output_processed)
        relative_std = stats["std"] / signal_std if signal_std > 0 else np.inf

        _, std_threshold = self._get_thresholds(test_case)

        assert relative_std < std_threshold, (
            f"Test case {test_case.name}: "
            f"Relative std {relative_std:.6f} above threshold {std_threshold}"
            f"{' (resampling involved)' if self._involves_resampling(test_case) else ''}"
        )

    def test_leveled_output_correlation(self, test_case: MatlabTestCase):
        """Test that leveled output has high correlation with MATLAB."""
        if test_case.output_leveled is None:
            pytest.skip("No leveled output in this test case")

        _, python_leveled = run_python_preprocessing(
            test_case.input_map,
            test_case.input_pixel_spacing,
            test_case.params,
            use_circle_center=bool(test_case.params.get("bMapCenterCircle", 0)),
        )

        if python_leveled is None:
            pytest.skip("Python implementation did not return leveled output")

        python_leveled, matlab_leveled = _crop_to_common_shape(
            python_leveled, test_case.output_leveled
        )
        correlation = _compute_correlation(python_leveled, matlab_leveled)
        corr_threshold, _ = self._get_thresholds(test_case)

        assert correlation > corr_threshold, (
            f"Test case {test_case.name}: "
            f"Leveled correlation {correlation:.6f} below threshold {corr_threshold}"
            f"{' (resampling involved)' if self._involves_resampling(test_case) else ''}"
        )

    def test_output_shape(self, test_case: MatlabTestCase):
        """Test that output shape matches MATLAB within 1 pixel."""
        python_processed, _ = run_python_preprocessing(
            test_case.input_map,
            test_case.input_pixel_spacing,
            test_case.params,
            use_circle_center=bool(test_case.params.get("bMapCenterCircle", 0)),
        )

        matlab_shape = test_case.output_processed.shape
        python_shape = python_processed.shape

        row_diff = abs(matlab_shape[0] - python_shape[0])
        col_diff = abs(matlab_shape[1] - python_shape[1])

        assert row_diff <= 1, (
            f"Test case {test_case.name}: "
            f"Row difference {row_diff} > 1 (MATLAB: {matlab_shape[0]}, Python: {python_shape[0]})"
        )
        assert col_diff <= 1, (
            f"Test case {test_case.name}: "
            f"Column difference {col_diff} > 1 (MATLAB: {matlab_shape[1]}, Python: {python_shape[1]})"
        )
