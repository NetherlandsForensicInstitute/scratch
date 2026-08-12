from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.constants import micro
from skimage.transform import resize

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.resample import (
    _clip_factors,
    _resample_scan_image,
    get_scaling_factors,
    resample_array_2d,
    resample_mark,
    resample_nan_aware,
    resample_scan_image_and_mask,
    resize_nan_aware,
)


class TestGetScalingFactors:
    def test_basic_calculation(self):
        assert get_scaling_factors((2 * micro, 2 * micro), 4 * micro) == (2.0, 2.0)

    def test_different_axes(self):
        assert get_scaling_factors((micro, 2 * micro), 4 * micro) == (4.0, 2.0)

    def test_upsampling(self):
        assert get_scaling_factors((8 * micro, 8 * micro), 4 * micro) == (0.5, 0.5)


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
    def test_keeps_the_output_shape_fractional(self):
        # skimage divides the input shape by the output shape to get its sampling step and
        # anti-aliasing sigma, rounding only when it allocates the result. Handing it a pre-rounded
        # (33, 67) would resample a 200 px axis at a step of 2.985 instead of the requested 3.0.
        with patch("conversion.resample.resize") as mock:
            mock.return_value = np.zeros((33, 67))

            resample_array_2d(np.zeros((100, 200)), factors=(3.0, 3.0))

            assert mock.call_args.kwargs["output_shape"] == pytest.approx(
                (100 / 3, 200 / 3)
            )
            assert mock.call_args.kwargs["anti_aliasing"] is True

    def test_matches_an_unrounded_skimage_resize(self):
        # Guards the whole path end-to-end, not just the argument: the values have to land where a
        # direct skimage call would put them. This is the behavior every non-CMC pipeline relies on.
        array = np.arange(100 * 200, dtype=np.float64).reshape(100, 200)
        factors = (3.0, 3.0)

        result = resample_array_2d(array, factors=factors)

        expected = resize(
            image=array,
            output_shape=(1 / factors[1] * 100, 1 / factors[0] * 200),
            mode="edge",
            anti_aliasing=True,
        )
        assert np.array_equal(result, expected)

    def test_spreads_nan(self):
        # Arrange: a single missing pixel, which resample_nan_aware would average away.
        array = np.ones((4, 4))
        array[0, 0] = np.nan

        # Act
        result = resample_array_2d(array, factors=(2.0, 2.0))

        # Assert
        assert np.isnan(result[0, 0])


class TestResampleNanAware:
    @patch("conversion.resample.resize_nan_aware")
    def test_calculates_output_shape_correctly(self, mock_resize: MagicMock):
        array = np.zeros((100, 200))
        mock_resize.return_value = np.zeros((50, 100))

        resample_nan_aware(array, factors=(2.0, 2.0))

        call_args = mock_resize.call_args.args
        assert call_args[1] == (50.0, 100.0)

    def test_averages_missing_data_away(self):
        # Arrange: a single missing pixel, which resample_array_2d would spread.
        array = np.ones((4, 4))
        array[0, 0] = np.nan

        # Act
        result = resample_nan_aware(array, factors=(2.0, 2.0))

        # Assert
        assert result[0, 0] == pytest.approx(1.0)

    def test_upsamples_without_degenerating_to_nearest_neighbor(self):
        # cv2 turns INTER_AREA into nearest-neighbor when zooming in, which would leave the output
        # holding only values already present in the input.
        array = np.arange(16, dtype=np.float64).reshape(4, 4)

        result = resample_nan_aware(array, factors=(0.5, 0.5), interpolation="linear")

        assert result.shape == (8, 8)
        assert not np.all(np.isin(result, array))

    def test_upsampling_keeps_a_hole_in_place(self):
        # Arrange: a single missing pixel in an otherwise valid array, doubled in size.
        array = np.ones((8, 8))
        array[4, 4] = np.nan

        # Act
        result = resample_nan_aware(array, factors=(0.5, 0.5), interpolation="linear")

        # Assert: the hole neither vanishes nor grows beyond the source pixel it came from.
        missing = np.isnan(result)
        assert missing.any()
        assert missing.sum() <= 4
        assert result[0, 0] == pytest.approx(1.0)


class TestResizeNanAware:
    @staticmethod
    def _shrink(image, factor):
        height, width = image.shape
        target = (int(np.ceil(height / factor)), int(np.ceil(width / factor)))
        return resize_nan_aware(image, target, interpolation="area")

    def test_rejects_an_unknown_interpolation(self):
        with pytest.raises(ValueError, match="Unknown interpolation"):
            resize_nan_aware(np.zeros((4, 4)), (2, 2), interpolation="bogus")  # type: ignore[arg-type]

    def test_computes_block_mean(self):
        # Arrange: each 2x2 block holds a single distinct value.
        image = np.repeat(np.repeat(np.array([[1.0, 2.0], [3.0, 4.0]]), 2, 0), 2, 1)

        # Act
        result = self._shrink(image, 2)

        # Assert
        assert result == pytest.approx(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_averages_only_valid_pixels(self):
        # Arrange: one block is half missing; its mean must come from the survivors alone.
        image = np.array([[4.0, 4.0, 1.0, 1.0], [4.0, 4.0, np.nan, np.nan]])

        # Act
        result = self._shrink(image, 2)

        # Assert
        assert result[0, 0] == pytest.approx(4.0)
        assert result[0, 1] == pytest.approx(1.0)

    def test_block_below_validity_threshold_becomes_nan(self):
        # Arrange: three of four sub-pixels missing, i.e. 25% valid.
        image = np.array([[7.0, np.nan], [np.nan, np.nan]])

        # Act
        result = self._shrink(image, 2)

        # Assert
        assert np.isnan(result[0, 0])

    def test_fully_missing_block_becomes_nan_without_warning(self):
        # Arrange
        image = np.full((2, 2), np.nan)

        # Act
        with np.errstate(all="raise"):
            result = self._shrink(image, 2)

        # Assert
        assert np.isnan(result).all()


class TestResampleScanImage:
    def test_updates_scales(self, scan_image_rectangular_with_nans: ScanImage):
        with patch("conversion.resample.resample_array_2d") as mock:
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

    def test_resamples_a_scale_mismatch_numpy_defaults_would_skip(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        # A 3.00003e-6 versus 3e-6 pixel size is a real mismatch, but sits inside numpy's own 1e-5
        # relative tolerance.
        result, _ = resample_scan_image_and_mask(
            scan_image_rectangular_with_nans, factors=(1.00001, 1.00001)
        )

        assert result is not scan_image_rectangular_with_nans

    def test_skips_a_scale_difference_that_is_only_float_rounding(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        result, _ = resample_scan_image_and_mask(
            scan_image_rectangular_with_nans, factors=(1.0000000000033, 1.0)
        )

        assert result is scan_image_rectangular_with_nans

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
                scan_image_rectangular_with_nans, target_scale=4 * micro
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

        with patch("conversion.resample.resample_array_2d") as mock:
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
    def test_uses_mark_target_sampling(self, impression_mark: Mark):
        resampled = resample_mark(impression_mark)

        scale = impression_mark.mark_type.scale
        assert resampled.scan_image.scale_x == scale
        assert resampled.scan_image.scale_y == scale
