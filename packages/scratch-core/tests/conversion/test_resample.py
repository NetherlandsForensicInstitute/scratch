import numpy as np
import pytest

from conversion.resample import get_resampling_factors, resample_image_and_mask
from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


class TestGetResamplingFactors:
    """Tests for get_resampling_factors function."""

    def test_resample_factor_overrides_target_resolution(self):
        factor_x, factor_y = get_resampling_factors(
            10e-6, 10e-6, False, 2.0, 5e-6, False
        )
        assert factor_x == pytest.approx(0.5)
        assert factor_y == pytest.approx(0.5)

    def test_resample_factor_upsample(self):
        factor_x, factor_y = get_resampling_factors(
            10e-6, 10e-6, False, 0.5, 5e-6, False
        )
        assert factor_x == pytest.approx(2.0)
        assert factor_y == pytest.approx(2.0)

    def test_upsample_with_scales_bigger_than_target(self):
        factor_x, factor_y = get_resampling_factors(
            10e-6, 10e-6, False, None, 5e-6, False
        )
        assert factor_x == pytest.approx(2.0)
        assert factor_y == pytest.approx(2.0)

    def test_downsample_with_scales_smaller_than_target(self):
        factor_x, factor_y = get_resampling_factors(
            1e-6, 1e-6, False, None, 4e-6, False
        )
        assert factor_x == pytest.approx(0.25)
        assert factor_y == pytest.approx(0.25)

    def test_no_resampling_needed(self):
        factor_x, factor_y = get_resampling_factors(
            4e-6, 4e-6, False, None, 4e-6, False
        )
        assert factor_x == pytest.approx(1.0)
        assert factor_y == pytest.approx(1.0)

    def test_different_scales_lead_to_different_factors(self):
        factor_x, factor_y = get_resampling_factors(
            2e-6, 4e-6, False, None, 4e-6, False
        )
        assert factor_x == pytest.approx(0.5)
        assert factor_y == pytest.approx(1.0)

    def test_only_downsample_clamps_factors_to_1(self):
        factor_x, factor_y = get_resampling_factors(
            10e-6, 10e-6, True, None, 5e-6, False
        )
        assert factor_x == pytest.approx(1.0)
        assert factor_y == pytest.approx(1.0)

    def test_only_downsample_preserves_factors_below_1(self):
        factor_x, factor_y = get_resampling_factors(1e-6, 1e-6, True, None, 4e-6, False)
        assert factor_x == pytest.approx(0.25)
        assert factor_y == pytest.approx(0.25)

    def test_only_downsample_clips_factors_above_1(self):
        factor_x, factor_y = get_resampling_factors(8e-6, 2e-6, True, None, 4e-6, False)
        assert factor_x == pytest.approx(1.0)
        assert factor_y == pytest.approx(0.5)

    def test_only_downsample_clamps_resample_factor_parameter(self):
        factor_x, factor_y = get_resampling_factors(
            10e-6, 10e-6, True, 0.5, 5e-6, False
        )
        assert factor_x == pytest.approx(1.0)
        assert factor_y == pytest.approx(1.0)

    def test_preserve_aspect_ratio_with_upsample(self):
        """Test that aspect ratio is preserved when upsampling."""
        factor_x, factor_y = get_resampling_factors(
            20e-6, 10e-6, False, None, 5e-6, True
        )
        assert factor_x == pytest.approx(factor_y)
        assert factor_x > 1.0

    def test_preserve_aspect_ratio_with_downsample(self):
        """Test that aspect ratio is preserved when downsampling."""
        factor_x, factor_y = get_resampling_factors(
            20e-6, 2e-6, False, None, 5e-6, True
        )
        assert factor_x == pytest.approx(factor_y)
        assert factor_x < 1.0


class TestResample:
    """Tests for resample function."""

    def test_output_shape_matches_resample_size(self, scan_image: ScanImage):
        """Output array shape matches original shape."""
        result, _ = resample_image_and_mask(scan_image, target_resolution=4e-6)
        assert result.data.shape == scan_image.data.shape

    def test_output_shape_matches_clamped_upsampled_size(self, scan_image: ScanImage):
        """Output array shape matches expected size (unchanged since only_downsample is True)."""
        result, _ = resample_image_and_mask(
            scan_image, target_resolution=1e-6, only_downsample=True
        )
        assert result.data.shape == scan_image.data.shape

    def test_output_shape_matches_upsampled_size(self, scan_image: ScanImage):
        """Output array shape matches expected upsampled size."""
        result, _ = resample_image_and_mask(
            scan_image, target_resolution=1e-6, only_downsample=False
        )
        assert result.data.shape == tuple(i * 4 for i in scan_image.data.shape)

    def test_scale_updated_according_to_target_resolution(self, scan_image: ScanImage):
        """Output scales are updated correctly."""
        result, _ = resample_image_and_mask(scan_image, target_resolution=8e-6)
        assert result.scale_x == pytest.approx(8e-6)
        assert result.scale_y == pytest.approx(8e-6)

    def test_no_resampling_returns_original(
        self, scan_image: ScanImage, mask_array: MaskArray
    ):
        """When no resampling needed, returns original objects."""
        result, result_mask = resample_image_and_mask(
            scan_image,
            mask=mask_array,
            target_resolution=0.5e-6,
            only_downsample=True,
        )
        assert result is scan_image
        assert result_mask is mask_array

    def test_mask_none_passthrough(self, scan_image: ScanImage):
        """When mask is None, returns None."""
        _, result_mask = resample_image_and_mask(
            scan_image, mask=None, target_resolution=4e-6
        )
        assert result_mask is None

    def test_mask_resampled_to_same_shape_as_image(
        self, scan_image: ScanImage, mask_array: MaskArray
    ):
        """Mask is resampled to same shape as image."""
        result, result_mask = resample_image_and_mask(
            scan_image, mask=mask_array, target_resolution=1e-6
        )
        assert result_mask is not None
        assert result_mask.shape == result.data.shape

    def test_mask_stays_binary(self, scan_image: ScanImage, mask_array: MaskArray):
        """Mask values remain binary after resampling."""
        _, result_mask = resample_image_and_mask(
            scan_image, mask=mask_array, target_resolution=1e-6
        )
        assert result_mask is not None
        unique_values = np.unique(result_mask)
        assert all(v in [0, 1] for v in unique_values)
