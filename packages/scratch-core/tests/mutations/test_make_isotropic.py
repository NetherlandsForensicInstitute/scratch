import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture

from container_models.scan_image import ScanImage
from mutations.spatial import MakeIsotropic


class TestMakeIsotropic:
    def test_make_isotropic_no_op(self):
        """Ensure no resampling occurs if pixels are already square."""
        scale = 0.5
        width, height = 100, 100
        scan_image = ScanImage(
            scale_x=scale, scale_y=scale, data=np.zeros((height, width))
        )

        result = MakeIsotropic()(scan_image)

        assert np.isclose(result.scale_x, scale), (
            f"Scale should not have changed, but now is {result.scale_x}"
        )
        assert np.isclose(result.scale_y, scale), (
            f"Scale should not have changed, but now is {result.scale_y}"
        )
        assert result.data.shape == (height, width), (
            f"Shape should not have changed, but now is {result.data.shape}"
        )
        np.testing.assert_array_equal(result.data, scan_image.data)

    def test_make_isotropic_upsampling_logic(self):
        """Verify upsampling to the smallest scale and correct shape calculation."""
        width, height = 100, 100
        scale_coarse = 2.0
        scale_fine = 1.0  # target scale
        scan_image = ScanImage(
            data=np.zeros((height, width)),
            scale_x=scale_coarse,
            scale_y=scale_fine,
        )

        result = MakeIsotropic()(scan_image)

        assert np.isclose(result.scale_x, scale_fine), (
            f"Scale should now be the minimum of the two {scale_fine}"
        )
        assert np.isclose(result.scale_y, scale_fine), (
            f"Scale should now be the minimum of the two {scale_fine}"
        )
        assert result.data.shape == (height, width * scale_coarse / scale_fine), (
            f"New width should be (original_width * (original_scale_x / target_scale)), but got: {result.data.shape}"
        )

    def test_make_isotropic_preserves_metadata_and_range(self):
        """Ensure metadata is passed through and pixel intensities remain consistent."""
        scan_image = ScanImage(
            meta_data={"sensor_id": "XY-Z", "timestamp": "2026-01-21"},
            scale_x=1.0,
            scale_y=0.5,
            data=np.array([[0, 1000], [2000, 3000]], dtype=np.float64),
        )

        result = MakeIsotropic()(scan_image)

        assert result.meta_data == scan_image.meta_data
        assert np.min(result.data) == 0, (
            f"Pixel intensity range should be preserved, but lowest value now is {np.min(result.data)}"
        )
        assert np.max(result.data) == 3000, (
            f"Pixel intensity range should be preserved, but highest value now is {np.max(result.data)}"
        )
        assert result.data.dtype == scan_image.data.dtype

    @pytest.mark.parametrize("scaling_factor", [4, 7.6, 8.1, 10.11])
    def test_make_isotropic_handles_nans(
        self, scan_image_rectangular_with_nans: ScanImage, scaling_factor: float
    ):
        """Ensure the resampling deals with NaN values correctly."""
        scale_fine = 1.5
        scale_coarse = 1.5 * scaling_factor
        scan_image = ScanImage(
            data=scan_image_rectangular_with_nans.data,
            scale_x=scale_fine,
            scale_y=scale_coarse,
        )
        expected_shape = (
            int(round(scan_image.height * scaling_factor)),
            int(round(scan_image.width)),
        )

        result = MakeIsotropic()(scan_image)

        assert np.isclose(result.scale_x, scale_fine), (
            f"Scale should be {scale_fine}, but got {result.scale_x}"
        )
        assert np.isclose(result.scale_y, scale_fine), (
            f"Scale should be {scale_fine}, but got {result.scale_y}"
        )
        assert result.data.shape == expected_shape, (
            f"Shape should be {expected_shape}, but got {result.data.shape}"
        )
        assert result.valid_mask.sum() / scan_image.valid_mask.sum() == pytest.approx(
            scaling_factor, abs=1e-3
        ), "The number of valid pixels / NaNs have not scaled correctly"

    def test_isotropic_skips_when_already_isotropic(
        self, caplog: LogCaptureFixture
    ) -> None:
        # Arrange
        shape = (4, 4)
        scan_image = ScanImage(
            data=np.ones(shape, dtype=float),
            scale_x=1.0,
            scale_y=1.0,
        )
        # Act
        resulting_image = MakeIsotropic()(scan_image)

        # Assert
        assert (
            f"Image is already isotropic, with shape: {shape}, conversion not needed."
            in caplog.messages
        ), (
            "A log message should have been recorded for skipping the isotropic mutation."
        )
        assert id(scan_image) == id(resulting_image), (
            "Should have returned the same scan_image."
        )
