import numpy as np
import pytest
from numpy.typing import NDArray

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.get_cropped_image import get_cropped_image
from conversion.leveling import SurfaceTerms
from pathlib import Path


@pytest.mark.integration
@pytest.mark.parametrize(
    "terms, regression_order",
    [
        (SurfaceTerms.PLANE, 0),
        (SurfaceTerms.PLANE, 1),
        (SurfaceTerms.PLANE, 2),
        (SurfaceTerms.SPHERE, 0),
        (SurfaceTerms.SPHERE, 1),
        (SurfaceTerms.SPHERE, 2),
    ],
)
def test_get_cropped_image(
    scan_image_replica: ScanImage,
    mask_array: MaskArray,
    terms: SurfaceTerms,
    regression_order: int,
):
    result = get_cropped_image(
        scan_image=scan_image_replica,
        mask=mask_array,
        terms=terms,
        regression_order=regression_order,
        cutoff_length=250e-6,
        resampling_factors=(2, 2),
    )
    assert result.shape == (mask_array.shape[0] // 2, mask_array.shape[1] // 2)


class TestGetCroppedImageMatlabComparison:
    """Test get_cropped_image against MATLAB reference output. Can be removed later."""

    @pytest.fixture
    def matlab_output_plane(self, baseline_images_dir: Path) -> np.ndarray:
        """Load MATLAB reference output from CSV file."""
        return np.load(
            baseline_images_dir / "get_cropped_image_output_plane_matlab.npy",
            allow_pickle=True,
        )

    @pytest.fixture
    def matlab_output_sphere(self, baseline_images_dir: Path) -> np.ndarray:
        """Load MATLAB reference output from CSV file."""
        return np.load(
            baseline_images_dir / "get_cropped_image_output_sphere_matlab.npy",
            allow_pickle=True,
        )

    @pytest.fixture
    def input_scan_image(self, baseline_images_dir: Path) -> "ScanImage":
        """Load input scan image used for MATLAB test."""
        data = np.load(
            baseline_images_dir / "get_cropped_image_input_image.npy", allow_pickle=True
        )
        xdim = 8.7479e-07  # meters
        ydim = 8.7479e-07  # meters
        return ScanImage(data=data, scale_x=xdim, scale_y=ydim)

    @pytest.fixture
    def input_mask(self, baseline_images_dir: Path) -> "MaskArray":
        """Create mask for input data (True for valid pixels)."""
        return np.load(
            baseline_images_dir / "get_cropped_image_input_mask.npy", allow_pickle=True
        ).astype(bool)

    @pytest.mark.integration
    def test_correlation_with_matlab_plane(
        self,
        matlab_output_plane: NDArray[np.floating],
        input_scan_image: ScanImage,
        input_mask: MaskArray,
    ):
        """
        Test that Python output correlates highly with MATLAB output.

        MATLAB parameters used:
        - level_method: 'Plane'
        - filter_method: 'R2' (regression order 2)
        - sampling: 4
        - cutoff_hi: 250 (micrometers)
        """
        python_output = get_cropped_image(
            scan_image=input_scan_image,
            mask=input_mask,
            terms=SurfaceTerms.PLANE,
            cutoff_length=250e-6,
            resampling_factors=(4, 4),
            regression_order=2,
        )

        assert python_output.shape == matlab_output_plane.shape, (
            f"Shape mismatch: Python {python_output.shape} vs MATLAB {matlab_output_plane.shape}"
        )

        # Compute correlation on valid (non-NaN) pixels
        valid_mask = ~(np.isnan(matlab_output_plane) | np.isnan(python_output))
        correlation = np.corrcoef(
            matlab_output_plane[valid_mask], python_output[valid_mask]
        )[0, 1]

        assert correlation > 0.99, (
            f"Correlation {correlation:.6f} is below threshold 0.99"
        )

    @pytest.mark.integration
    def test_correlation_with_matlab_sphere(
        self,
        matlab_output_sphere: NDArray[np.floating],
        input_scan_image: ScanImage,
        input_mask: MaskArray,
    ):
        """
        Test that Python output correlates highly with MATLAB output.

        MATLAB parameters used:
        - level_method: 'Plane'
        - filter_method: 'R2' (regression order 2)
        - sampling: 4
        - cutoff_hi: 250 (micrometers)
        """
        python_output = get_cropped_image(
            scan_image=input_scan_image,
            mask=input_mask,
            terms=SurfaceTerms.SPHERE,
            cutoff_length=250e-6,
            resampling_factors=(4, 4),
            regression_order=2,
        )

        assert python_output.shape == matlab_output_sphere.shape, (
            f"Shape mismatch: Python {python_output.shape} vs MATLAB {matlab_output_sphere.shape}"
        )

        # Compute correlation on valid (non-NaN) pixels
        valid_mask = ~(np.isnan(matlab_output_sphere) | np.isnan(python_output))
        correlation = np.corrcoef(
            matlab_output_sphere[valid_mask], python_output[valid_mask]
        )[0, 1]

        assert correlation > 0.99, (
            f"Correlation {correlation:.6f} is below threshold 0.99"
        )

    @pytest.mark.integration
    def test_statistics_match_matlab_plane(
        self,
        matlab_output_plane: NDArray[np.floating],
        input_scan_image: ScanImage,
        input_mask: MaskArray,
    ) -> None:
        """Test that Python output statistics are close to MATLAB."""
        python_output = get_cropped_image(
            scan_image=input_scan_image,
            mask=input_mask,
            terms=SurfaceTerms.PLANE,
            cutoff_length=250e-6,
            resampling_factors=(4, 4),
            regression_order=2,
        )

        matlab_std = np.nanstd(matlab_output_plane)
        python_std = np.nanstd(python_output)
        std_ratio = python_std / matlab_std

        # Standard deviation should be within 5% of MATLAB
        assert 0.95 < std_ratio < 1.05, (
            f"Std ratio {std_ratio:.4f} outside tolerance: "
            f"Python {python_std:.6e} vs MATLAB {matlab_std:.6e}"
        )

        matlab_min = np.nanmin(matlab_output_plane)
        matlab_max = np.nanmax(matlab_output_plane)
        python_min = np.nanmin(python_output)
        python_max = np.nanmax(python_output)

        # Min/max should be within 10% of MATLAB
        min_ratio = python_min / matlab_min
        max_ratio = python_max / matlab_max

        assert 0.9 < min_ratio < 1.1, (
            f"Min ratio {min_ratio:.4f} outside tolerance: "
            f"Python {python_min:.6e} vs MATLAB {matlab_min:.6e}"
        )
        assert 0.9 < max_ratio < 1.1, (
            f"Max ratio {max_ratio:.4f} outside tolerance: "
            f"Python {python_max:.6e} vs MATLAB {matlab_max:.6e}"
        )

    @pytest.mark.integration
    def test_statistics_match_matlab_sphere(
        self,
        matlab_output_sphere: NDArray[np.floating],
        input_scan_image: ScanImage,
        input_mask: MaskArray,
    ) -> None:
        """Test that Python output statistics are close to MATLAB."""
        python_output = get_cropped_image(
            scan_image=input_scan_image,
            mask=input_mask,
            terms=SurfaceTerms.SPHERE,
            cutoff_length=250e-6,
            resampling_factors=(4, 4),
            regression_order=2,
        )

        matlab_std = np.nanstd(matlab_output_sphere)
        python_std = np.nanstd(python_output)
        std_ratio = python_std / matlab_std

        # Standard deviation should be within 5% of MATLAB
        assert 0.95 < std_ratio < 1.05, (
            f"Std ratio {std_ratio:.4f} outside tolerance: "
            f"Python {python_std:.6e} vs MATLAB {matlab_std:.6e}"
        )

        matlab_min = np.nanmin(matlab_output_sphere)
        matlab_max = np.nanmax(matlab_output_sphere)
        python_min = np.nanmin(python_output)
        python_max = np.nanmax(python_output)

        # Min/max should be within 10% of MATLAB
        min_ratio = python_min / matlab_min
        max_ratio = python_max / matlab_max

        assert 0.9 < min_ratio < 1.1, (
            f"Min ratio {min_ratio:.4f} outside tolerance: "
            f"Python {python_min:.6e} vs MATLAB {matlab_min:.6e}"
        )
        assert 0.9 < max_ratio < 1.1, (
            f"Max ratio {max_ratio:.4f} outside tolerance: "
            f"Python {python_max:.6e} vs MATLAB {matlab_max:.6e}"
        )
