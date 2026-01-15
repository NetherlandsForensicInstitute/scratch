import numpy as np
from unittest.mock import patch, MagicMock

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.resample import (
    resample_scan_image_and_mask,
    _resample_scan_image,
    get_scaling_factors,
    _clip_factors,
    resample_image_array,
    resample_mark,
)


class TestGetScalingFactors:
    def test_basic_calculation(self):
        assert get_scaling_factors((2e-6, 2e-6), 4e-6) == (2.0, 2.0)

    def test_different_axes(self):
        assert get_scaling_factors((1e-6, 2e-6), 4e-6) == (4.0, 2.0)

    def test_upsampling(self):
        assert get_scaling_factors((8e-6, 8e-6), 4e-6) == (0.5, 0.5)


class TestClipFactors:
    def test_no_clipping_needed(self):
        assert _clip_factors((2.0, 1.5), False) == (2.0, 1.5)

    def test_clip_below_one(self):
        assert _clip_factors((0.5, 2.0), False) == (1.0, 2.0)

    def test_preserve_aspect_ratio_clips_to_max(self):
        assert _clip_factors((0.5, 2.0), True) == (2.0, 2.0)

    def test_preserve_aspect_ratio_all_below_one(self):
        assert _clip_factors((0.5, 0.8), True) == (1.0, 1.0)


class TestResampleArray:
    @patch("conversion.resample.resize")
    def test_calculates_output_shape_correctly(self, mock_resize: MagicMock):
        array = np.zeros((100, 200))
        mock_resize.return_value = np.zeros((50, 100))

        resample_image_array(array, factors=(2.0, 2.0))

        call_args = mock_resize.call_args[1]
        assert call_args["output_shape"] == (50.0, 100.0)

    @patch("conversion.resample.resize")
    def test_anti_aliasing_on_upsampling(self, mock_resize: MagicMock):
        array = np.zeros((100, 100))
        mock_resize.return_value = np.zeros((200, 200))

        resample_image_array(array, factors=(0.5, 0.5))

        assert mock_resize.call_args[1]["anti_aliasing"] is False

    @patch("conversion.resample.resize")
    def test_no_anti_aliasing_on_downsampling(self, mock_resize: MagicMock):
        array = np.zeros((100, 100))
        mock_resize.return_value = np.zeros((50, 50))

        resample_image_array(array, factors=(2.0, 2.0))

        assert mock_resize.call_args[1]["anti_aliasing"] is True


class TestResampleScanImage:
    def test_updates_scales(self, scan_image_rectangular_with_nans: ScanImage):
        with patch("conversion.resample.resample_image_array") as mock:
            mock.return_value = np.zeros((50, 50))

            result = _resample_scan_image(scan_image_rectangular_with_nans, (2.0, 2.0))

            assert result.scale_x == scan_image_rectangular_with_nans.scale_x * 2.0
            assert result.scale_y == scan_image_rectangular_with_nans.scale_y * 2.0


class TestResampleImageAndMask:
    def test_no_resampling_when_factors_close_to_one(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        mask = np.ones((100, 100), dtype=np.bool_)

        result_img, result_mask = resample_scan_image_and_mask(
            scan_image_rectangular_with_nans, mask, factors=(1.0, 1.0)
        )

        assert result_img is scan_image_rectangular_with_nans
        assert result_mask is mask

    def test_uses_explicit_factors(self, scan_image_rectangular_with_nans: ScanImage):
        with patch("conversion.resample.get_scaling_factors") as mock:
            resample_scan_image_and_mask(
                scan_image_rectangular_with_nans, factors=(2.0, 2.0)
            )
            mock.assert_not_called()

    def test_calculates_factors_when_not_provided(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        with patch("conversion.resample.get_scaling_factors") as mock:
            mock.return_value = (2.0, 2.0)
            resample_scan_image_and_mask(
                scan_image_rectangular_with_nans, target_scale=4e-6
            )
            mock.assert_called_once()

    def test_clips_when_only_downsample(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        with patch("conversion.resample._clip_factors") as mock_clip:
            mock_clip.return_value = (1.0, 1.0)
            resample_scan_image_and_mask(
                scan_image_rectangular_with_nans,
                factors=(0.5, 0.5),
                only_downsample=True,
            )
            mock_clip.assert_called_once_with((0.5, 0.5), True)

    def test_no_clip_when_only_downsample_false(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        with patch("conversion.resample._clip_factors") as mock_clip:
            with patch("conversion.resample._resample_scan_image"):
                resample_scan_image_and_mask(
                    scan_image_rectangular_with_nans,
                    factors=(0.5, 0.5),
                    only_downsample=False,
                )
                mock_clip.assert_not_called()

    def test_resamples_mask_when_provided(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        mask = np.ones((100, 100), dtype=np.bool_)

        with patch("conversion.resample.resample_image_array") as mock:
            mock.return_value = np.zeros((50, 50))

            _, result_mask = resample_scan_image_and_mask(
                scan_image_rectangular_with_nans, mask, factors=(2.0, 2.0)
            )

            assert mock.call_count == 2  # Once for image, once for mask

    def test_none_mask_stays_none(self, scan_image_rectangular_with_nans):
        _, result_mask = resample_scan_image_and_mask(
            scan_image_rectangular_with_nans, mask=None, factors=(2.0, 2.0)
        )

        assert result_mask is None


class TestResampleMark:
    def test_uses_mark_target_sampling(self, mark: Mark):
        resampled = resample_mark(mark)

        scale = mark.mark_type.scale
        assert resampled.scan_image.scale_x == scale
        assert resampled.scan_image.scale_y == scale
