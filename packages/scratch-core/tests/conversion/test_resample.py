import numpy as np
import pytest

from conversion.resample import get_resampling_factors, resample
from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


class TestGetResamplingFactors:
    """Tests for get_resampling_factors function."""

    @pytest.mark.parametrize(
        "scale_x, scale_y, only_downsample, sampling, target_sampling_distance, expected_x, expected_y",
        [
            pytest.param(
                10e-6,
                10e-6,
                False,
                2.0,
                5e-6,
                0.5,
                0.5,
                id="sampling overrides target_sampling_distance",
            ),
            pytest.param(
                10e-6, 10e-6, False, 0.5, 5e-6, 2.0, 2.0, id="sampling_upsample"
            ),
            pytest.param(
                10e-6,
                10e-6,
                False,
                None,
                5e-6,
                2.0,
                2.0,
                id="scales bigger than target_sampling_distance so upsample, factors bigger than one",
            ),
            pytest.param(
                1e-6,
                1e-6,
                False,
                None,
                4e-6,
                0.25,
                0.25,
                id="scales smaller than target_sampling_distance so downsample, factors smaller than one",
            ),
            pytest.param(
                4e-6, 4e-6, False, None, 4e-6, 1.0, 1.0, id="no resampling needed"
            ),
            pytest.param(
                2e-6,
                4e-6,
                False,
                None,
                4e-6,
                0.5,
                1.0,
                id="different scales lead to different factors",
            ),
            pytest.param(
                10e-6,
                10e-6,
                True,
                None,
                5e-6,
                1.0,
                1.0,
                id="only downsample so factors clipped to 1",
            ),
            pytest.param(
                1e-6,
                1e-6,
                True,
                None,
                4e-6,
                0.25,
                0.25,
                id="only downsample preserves factors below 1",
            ),
            pytest.param(
                8e-6,
                2e-6,
                True,
                None,
                4e-6,
                1.0,
                0.5,
                id="only downsample clippes only factors above 1",
            ),
            pytest.param(
                10e-6,
                10e-6,
                True,
                0.5,
                5e-6,
                1.0,
                1.0,
                id="only downsample also clamps samping parameter",
            ),
        ],
    )
    def test_resampling_factors(
        self,
        scale_x,
        scale_y,
        only_downsample,
        sampling,
        target_sampling_distance,
        expected_x,
        expected_y,
    ):
        """Test resampling factor calculation."""
        factor_x, factor_y = get_resampling_factors(
            scale_x, scale_y, only_downsample, sampling, target_sampling_distance
        )
        assert factor_x == pytest.approx(expected_x)
        assert factor_y == pytest.approx(expected_y)


class TestResample:
    """Tests for resample function."""

    def test_output_shape_matches_resample_size(self, scan_image: ScanImage):
        """Output array shape matches expected downsampled size."""
        result, _ = resample(scan_image, target_sampling_distance=4e-6)
        assert result.data.shape == scan_image.data.shape

    def test_output_shape_matches_clamped_upsampled_size(self, scan_image: ScanImage):
        """Output array shape matches expected downsampled size."""
        result, _ = resample(
            scan_image, target_sampling_distance=1e-6, only_downsample=True
        )
        assert result.data.shape == scan_image.data.shape

    def test_output_shape_matches_upsampled_size(self, scan_image: ScanImage):
        """Output array shape matches expected downsampled size."""
        result, _ = resample(
            scan_image, target_sampling_distance=1e-6, only_downsample=False
        )
        assert result.data.shape == tuple(i * 4 for i in scan_image.data.shape)

    def test_scale_updated_according_to_target_sampling_distance(
        self, scan_image: ScanImage
    ):
        """Output scales are updated correctly."""
        result, _ = resample(scan_image, target_sampling_distance=8e-6)
        assert result.scale_x == pytest.approx(8e-6)
        assert result.scale_y == pytest.approx(8e-6)

    def test_no_resampling_returns_original(
        self, scan_image: ScanImage, mask_array: MaskArray
    ):
        """When no resampling needed, returns original objects."""
        result, result_mask = resample(
            scan_image,
            mask=mask_array,
            target_sampling_distance=0.5e-6,
            only_downsample=True,
        )
        assert result is scan_image
        assert result_mask is mask_array

    def test_mask_none_passthrough(self, scan_image: ScanImage):
        """When mask is None, returns None."""
        _, result_mask = resample(scan_image, mask=None, target_sampling_distance=4e-6)
        assert result_mask is None

    def test_mask_resampled_to_same_shape_as_image(
        self, scan_image: ScanImage, mask_array: MaskArray
    ):
        """Mask is resampled to same shape as image."""
        result, result_mask = resample(
            scan_image, mask=mask_array, target_sampling_distance=1e-6
        )
        assert result_mask is not None
        assert result_mask.shape == result.data.shape

    def test_mask_stays_binary(self, scan_image: ScanImage, mask_array: MaskArray):
        """Mask values remain binary after resampling."""
        _, result_mask = resample(
            scan_image, mask=mask_array, target_sampling_distance=1e-6
        )
        assert result_mask is not None
        unique_values = np.unique(result_mask)
        assert all(v in [0, 1] for v in unique_values)
