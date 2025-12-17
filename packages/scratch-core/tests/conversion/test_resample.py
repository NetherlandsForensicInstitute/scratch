import numpy as np
import pytest

from conversion.resample import (
    get_resampling_factors,
    resample_image_and_mask,
    clip_resample_factors,
    resample_mark,
)
from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


class TestGetResamplingFactors:
    """Tests for get_resampling_factors function."""

    def test_upsample_with_scales_bigger_than_target(self):
        factor_x, factor_y = get_resampling_factors(10e-6, 10e-6, 5e-6)
        assert factor_x == pytest.approx(0.5)
        assert factor_y == pytest.approx(0.5)

    def test_downsample_with_scales_smaller_than_target(self):
        factor_x, factor_y = get_resampling_factors(1e-6, 1e-6, 4e-6)
        assert factor_x == pytest.approx(4.0)
        assert factor_y == pytest.approx(4.0)

    def test_no_resampling_needed(self):
        factor_x, factor_y = get_resampling_factors(4e-6, 4e-6, 4e-6)
        assert factor_x == pytest.approx(1.0)
        assert factor_y == pytest.approx(1.0)

    def test_different_scales_lead_to_different_factors(self):
        factor_x, factor_y = get_resampling_factors(2e-6, 4e-6, 4e-6)
        assert factor_x == pytest.approx(2.0)
        assert factor_y == pytest.approx(1.0)


class TestClipResampleFactors:
    """Tests for clip_resample_factors function."""

    def test_only_downsample_clamps_factors_below_1(self):
        """Factors below 1 (upsampling) get clamped to 1."""
        result = clip_resample_factors((0.5, 0.5), preserve_aspect_ratio=False)
        assert result == (1.0, 1.0)

    def test_only_downsample_preserves_factors_above_1(self):
        """Factors above 1 (downsampling) are preserved."""
        result = clip_resample_factors((4.0, 4.0), preserve_aspect_ratio=False)
        assert result == (4.0, 4.0)

    def test_only_downsample_clips_mixed_factors(self):
        """Mixed factors: those below 1 get clamped, those above 1 preserved."""
        result = clip_resample_factors((0.5, 2.0), preserve_aspect_ratio=False)
        assert result == (1.0, 2.0)

    def test_preserve_aspect_ratio_with_only_downsample(self):
        """Aspect ratio preserved first, then clamped if needed."""
        result = clip_resample_factors((0.5, 2.0), preserve_aspect_ratio=True)
        assert result == (2.0, 2.0)


class TestResample:
    """Tests for resample function."""

    def test_output_shape_matches_resample_size(self, scan_image: ScanImage):
        """Output array shape matches original shape."""
        result, _ = resample_image_and_mask(scan_image, target_scale=4e-6)
        assert result.data.shape == scan_image.data.shape

    def test_output_shape_matches_clamped_upsampled_size(self, scan_image: ScanImage):
        """Output array shape matches expected size (unchanged since only_downsample is True)."""
        result, _ = resample_image_and_mask(
            scan_image, target_scale=1e-6, only_downsample=True
        )
        assert result.data.shape == scan_image.data.shape

    def test_output_shape_matches_upsampled_size(self, scan_image: ScanImage):
        """Output array shape matches expected upsampled size."""
        result, _ = resample_image_and_mask(
            scan_image, target_scale=1e-6, only_downsample=False
        )
        assert result.data.shape == tuple(i * 4 for i in scan_image.data.shape)

    def test_scale_updated_according_to_target_scale(self, scan_image: ScanImage):
        """Output scales are updated correctly."""
        result, _ = resample_image_and_mask(scan_image, target_scale=8e-6)
        assert result.scale_x == pytest.approx(8e-6)
        assert result.scale_y == pytest.approx(8e-6)

    def test_no_resampling_returns_original(
        self, scan_image_replica: ScanImage, mask_array: MaskArray
    ):
        """When no resampling needed, returns original objects."""
        result, result_mask = resample_image_and_mask(
            scan_image_replica,
            mask=mask_array,
            target_scale=0.5e-6,
            only_downsample=True,
        )
        assert result is scan_image_replica
        assert result_mask is mask_array

    def test_mask_none_passthrough(self, scan_image: ScanImage):
        """When mask is None, returns None."""
        _, result_mask = resample_image_and_mask(
            scan_image, mask=None, target_scale=4e-6
        )
        assert result_mask is None

    def test_mask_resampled_to_same_shape_as_image(
        self, scan_image_replica: ScanImage, mask_array: MaskArray
    ):
        """Mask is resampled to same shape as image."""
        result, result_mask = resample_image_and_mask(
            scan_image_replica, mask=mask_array, target_scale=1e-6
        )
        assert result_mask is not None
        assert result_mask.shape == result.data.shape

    def test_mask_stays_binary(
        self, scan_image_replica: ScanImage, mask_array: MaskArray
    ):
        """Mask values remain binary after resampling."""
        _, result_mask = resample_image_and_mask(
            scan_image_replica, mask=mask_array, target_scale=1e-6
        )
        assert result_mask is not None
        unique_values = np.unique(result_mask)
        assert all(v in [0, 1] for v in unique_values)


class TestResampleMark:
    def test_uses_mark_target_sampling(self, mark_image):
        resampled = resample_mark(mark_image)

        expected_sampling = mark_image.mark_type.sampling_rate
        assert resampled.scale_x == expected_sampling
        assert resampled.scale_y == expected_sampling
