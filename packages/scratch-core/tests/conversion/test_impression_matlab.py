"""
Tests comparing Python preprocess_impression output with MATLAB reference.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.preprocess_impression.impression import preprocess_impression_mark
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from .helper_functions import (
    _compute_correlation,
    _crop_to_common_shape,
    _compute_difference_stats,
)


@dataclass
class MatlabTestCase:
    """Container for a single MATLAB test case."""

    name: str
    # Input
    input_data: FloatArray2D
    pixel_spacing: tuple[float, float]
    # Processing options
    params: PreprocessingImpressionParams
    use_circle_center: bool
    # Expected output
    output_data: FloatArray2D
    output_leveled: FloatArray2D | None
    # Flags
    b_interpol: bool
    target_pixel_spacing: float | None

    @classmethod
    def from_directory(cls, case_dir: Path) -> "MatlabTestCase":
        """Load a test case from a directory."""
        with open(case_dir / "metadata.json") as f:
            meta = json.load(f)

        # Load arrays
        input_data = np.load(case_dir / "input_data.npy")
        output_data = np.load(case_dir / "output_data.npy")

        output_leveled = None
        if (
            meta.get("has_leveled_output")
            and (case_dir / "output_leveled.npy").exists()
        ):
            output_leveled = np.load(case_dir / "output_leveled.npy")

        # Parse pixel spacing
        ps = meta["pixel_spacing"]
        pixel_spacing = (float(ps[0]), float(ps[1]))

        # Parse target pixel spacing
        target_pixel_spacing = meta.get("target_pixel_spacing")

        # Convert level params to flags
        level_params = meta.get("level_params", [])
        level_offset = 1 in level_params
        level_tilt = 2 in level_params and 3 in level_params
        level_2nd = 4 in level_params and 5 in level_params and 6 in level_params

        # Convert filter params
        filters = meta.get("filters", [])
        highpass_cutoff = None
        lowpass_cutoff = None
        highpass_order = 2
        lowpass_order = 0

        for f in filters:
            cutoff = f.get("cutoff_length", [0, 0])
            cutoff_val = float(cutoff[0]) if isinstance(cutoff, list) else float(cutoff)

            if f.get("is_highpass", False):
                highpass_cutoff = cutoff_val
                highpass_order = int(f.get("n_order", 2))
            else:
                lowpass_cutoff = cutoff_val
                lowpass_order = int(f.get("n_order", 0))

        params = PreprocessingImpressionParams(
            adjust_pixel_spacing=meta.get("adjust_pixel_spacing", False),
            level_offset=level_offset,
            level_tilt=level_tilt,
            level_2nd=level_2nd,
            pixel_size=target_pixel_spacing,
            highpass_cutoff=highpass_cutoff,
            lowpass_cutoff=lowpass_cutoff,
            highpass_regression_order=highpass_order,
            lowpass_regression_order=lowpass_order,
        )

        return cls(
            name=case_dir.name,
            input_data=input_data,
            pixel_spacing=pixel_spacing,
            params=params,
            use_circle_center=meta.get("use_circle_center", False),
            output_data=output_data,
            output_leveled=output_leveled,
            b_interpol=meta.get("b_interpol", False),
            target_pixel_spacing=target_pixel_spacing,
        )

    @property
    def involves_resampling(self) -> bool:
        """Check if test case involves resampling."""
        if self.b_interpol:
            return True
        if self.target_pixel_spacing is None:
            return False
        return (
            abs(self.target_pixel_spacing - self.pixel_spacing[0]) > 1e-9
            or abs(self.target_pixel_spacing - self.pixel_spacing[1]) > 1e-9
        )

    @property
    def has_leveled_output(self) -> bool:
        """Check if test case has leveled output."""
        return self.output_leveled is not None


def discover_test_cases(test_cases_dir: Path) -> list[MatlabTestCase]:
    """Discover all test cases in a directory."""
    if not test_cases_dir.exists():
        return []
    return [
        MatlabTestCase.from_directory(d)
        for d in sorted(test_cases_dir.iterdir())
        if d.is_dir() and (d / "metadata.json").exists()
    ]


def run_python_preprocessing(
    test_case: MatlabTestCase,
) -> tuple[FloatArray2D, FloatArray2D | None]:
    """Run Python preprocessing and return (processed, leveled) arrays."""
    scan_image = ScanImage(
        data=test_case.input_data,
        scale_x=test_case.pixel_spacing[0],
        scale_y=test_case.pixel_spacing[1],
    )

    mark_type = (
        MarkType.BREECH_FACE_IMPRESSION
        if test_case.use_circle_center
        else MarkType.FIRING_PIN_IMPRESSION
    )
    mark = Mark(scan_image=scan_image, mark_type=mark_type)

    processed, leveled = preprocess_impression_mark(mark, test_case.params)

    return processed.scan_image.data, leveled.scan_image.data


@pytest.fixture(scope="module")
def test_cases_dir(baseline_images_dir) -> Path:
    return baseline_images_dir / "preprocess_impression"


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
        / "preprocess_impression"
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


class TestPreprocessImpressionMatlabComparison:
    """Test Python preprocess_impression against MATLAB reference outputs."""

    THRESHOLDS = {
        "default": (0.9999, 0.001),
        "resampled": (0.97, 0.2),
    }

    def _get_thresholds(self, test_case: MatlabTestCase) -> tuple[float, float]:
        """Get thresholds based on test case characteristics."""
        if test_case.involves_resampling:
            return self.THRESHOLDS["resampled"]
        return self.THRESHOLDS["default"]

    def test_matlab_comparison(self, test_case: MatlabTestCase):
        """Test that Python output matches MATLAB reference."""
        python_processed, python_leveled = run_python_preprocessing(test_case)
        matlab_processed = test_case.output_data

        # Shape check
        row_diff = abs(matlab_processed.shape[0] - python_processed.shape[0])
        col_diff = abs(matlab_processed.shape[1] - python_processed.shape[1])
        assert row_diff <= 1 and col_diff <= 1, (
            f"{test_case.name}: shape mismatch "
            f"{python_processed.shape} vs {matlab_processed.shape}"
        )

        corr_threshold, std_threshold = self._get_thresholds(test_case)
        suffix = "(resampled)" if test_case.involves_resampling else ""

        # Processed output checks
        python_cropped, matlab_cropped = _crop_to_common_shape(
            python_processed, matlab_processed
        )

        correlation = _compute_correlation(python_cropped, matlab_cropped)
        assert correlation > corr_threshold, (
            f"{test_case.name}: correlation {correlation:.4f} < {corr_threshold} {suffix}"
        )

        stats = _compute_difference_stats(python_cropped, matlab_cropped)
        signal_std = np.nanstd(matlab_processed)
        relative_std = stats["std"] / signal_std if signal_std > 0 else np.inf
        assert relative_std < std_threshold, (
            f"{test_case.name}: relative_std {relative_std:.4f} > {std_threshold} {suffix}"
        )

        # Leveled output checks (if available)
        if test_case.output_leveled is not None and python_leveled is not None:
            python_leveled_cropped, matlab_leveled_cropped = _crop_to_common_shape(
                python_leveled, test_case.output_leveled
            )

            leveled_corr = _compute_correlation(
                python_leveled_cropped, matlab_leveled_cropped
            )
            assert leveled_corr > corr_threshold, (
                f"{test_case.name}: leveled correlation {leveled_corr:.4f} < {corr_threshold} {suffix}"
            )
