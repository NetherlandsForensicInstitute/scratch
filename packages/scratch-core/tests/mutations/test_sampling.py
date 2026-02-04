import logging
from pathlib import Path

from math import ceil
import pytest
import numpy as np
from returns.pipeline import is_successful
from scipy.constants import milli
from container_models.base import Pair
from container_models.image import ImageContainer, MetaData
from mutations.sampling import IsotropicResample, Subsample
from ..helper_function import unwrap_result


class TestSubsampleMutation:
    # TODO: find a better test methology
    def test_subsample_matches_baseline_output(
        self, baseline_images_dir: Path, image_replica: ImageContainer
    ) -> None:
        # arrange
        verified = np.load(baseline_images_dir / "replica_subsampled.npy")
        subsample_scan_image = Subsample(step_size_x=10, step_size_y=15)
        # act - use copy to avoid mutating session-scoped fixture
        result = subsample_scan_image(image_replica)
        subsampled = unwrap_result(result)
        # assert
        #
        assert np.allclose(verified, subsampled.data, equal_nan=True)

    @pytest.mark.parametrize(
        "step_size_x, step_size_y",
        [(1, 1), (10, 10), (25, 25), (25, 50)],
    )
    def test_subsample_matches_size(
        self, image_container: ImageContainer, step_size_x: int, step_size_y: int
    ):
        # Arrange
        expected_height = ceil(image_container.height / step_size_y)
        expected_width = ceil(image_container.width / step_size_x)
        subsample_scan_image = Subsample(step_size_x, step_size_y)

        # Act
        result = subsample_scan_image(image_container)
        subsampled = unwrap_result(result)

        #  Assert
        assert subsampled.data.shape == (expected_height, expected_width)

    @pytest.mark.parametrize(
        ("step_x", "step_y"),
        [
            pytest.param(1, 1, id="default value"),
            pytest.param(10, 1, id="only x"),
            pytest.param(1, 10, id="only y"),
            pytest.param(10, 5, id="different x and y", marks=pytest.mark.xfail),
        ],
    )
    def test_subsample_updates_scan_image_scales(
        self, image_container: ImageContainer, step_x: int, step_y: int
    ) -> None:
        # Arrange
        subsample_scan_image = Subsample(step_x, step_y)

        # Act
        result = subsample_scan_image(image_container)
        subsampled = unwrap_result(result)

        # Assert
        assert np.isclose(
            subsampled.metadata.scale.x,
            image_container.metadata.scale.x * step_x,
            atol=milli,
        )
        assert np.isclose(
            subsampled.metadata.scale.y,
            image_container.metadata.scale.y * step_y,
            atol=milli,
        )

    @pytest.mark.parametrize(
        "step_size_x, step_size_y",
        [(-2, 2), (0, 0), (0, 3), (2, -1), (-1, -1), (1e3, 1e4)],
    )
    def test_subsample_rejects_incorrect_sizes(
        self, image_container: ImageContainer, step_size_x: int, step_size_y: int
    ):
        # Arrange
        subsample_scan_image = Subsample(step_size_x, step_size_y)
        # Act
        result = subsample_scan_image(image_container)

        # Assert
        assert not is_successful(result)

    def test_subsample_skips_when_given_step_size_of_one(
        self, image_container: ImageContainer, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Test when given the subsample the stepsize of one in both directions,
        it doesn't compute the whole image but just returns the original.
        """
        # Arrange
        subsample_scan_image = Subsample(1, 1)

        # Act
        with caplog.at_level(logging.INFO):
            result = subsample_scan_image(image_container)

        # Assert
        assert unwrap_result(result) is image_container, (
            "Expected the same object to be returned"
        )
        assert "No subsampling needed, returning original scan image" in caplog.text


class TestIsotropicResample:
    def test_make_isotropic_no_op(self):
        """Ensure no resampling occurs if pixels are already square."""
        # Arrange
        scale = Pair(0.5, 0.5)
        image = ImageContainer(
            data=np.zeros((100, 100)),
            metadata=MetaData(scale=scale),
        )
        # Act
        result = IsotropicResample(image)
        sample = unwrap_result(result)
        # Assert
        assert np.array_equal(sample.metadata.scale, scale), (
            f"Scale should not changed, {sample.metadata.scale}"
        )
        assert np.array_equal(sample.data, image.data), "Nothing should have changed"

    def test_make_isotropic_upsampling_logic(self):
        """Verify upsampling to the smallest scale and correct shape calculation."""
        # Arrange
        width = height = 100
        scale_coarse = 2.0
        scale_fine = 1.0  # target scale
        scale = Pair(scale_coarse, scale_fine)
        image = ImageContainer(
            data=np.zeros((100, 100)),
            metadata=MetaData(scale=scale),
        )
        # Act
        result = IsotropicResample(image)
        sample = unwrap_result(result)
        # Assert
        assert np.array_equal(sample.metadata.scale, (scale_fine, scale_fine)), (
            f"Scale should now be the minimum {scale_fine}"
        )
        assert (sample.height, sample.width) == (
            height,
            width * scale_coarse / scale_fine,
        ), (
            f"New width should be (original_width * (original_scale_x / target_scale)), but got: ({sample.height}, {sample.width})"
        )

    def test_make_isotropic_preserves_metadata_and_range(self):
        """Ensure metadata is passed through and pixel intensities remain consistent."""
        # Arrange
        image = ImageContainer(
            metadata=MetaData.model_validate(
                {"sensor_id": "XY-Z", "timestamp": "2026-01-21", "scale": (1, 0.5)}
            ),
            data=np.array([[0, 1000], [2000, 3000]], dtype=np.float64),
        )

        # Act
        result = IsotropicResample(image)
        sample = unwrap_result(result)
        # Assert
        assert sample.metadata.model_dump(
            exclude={"scale"}
        ) == image.metadata.model_dump(exclude={"scale"})
        assert np.min(sample.data) == 0, (
            f"Pixel intensity range should be preserved, but lowest value now is {np.min(sample.data)}"
        )
        assert np.max(sample.data) == 3000, (
            f"Pixel intensity range should be preserved, but highest value now is {np.max(sample.data)}"
        )
        assert sample.data.dtype == sample.data.dtype

    @pytest.mark.parametrize("scaling_factor", [4, 7.6, 8.1, 10.11])
    def test_make_isotropic_handles_nans(
        self, image_rectangular_with_nans: ImageContainer, scaling_factor: float
    ):
        """Ensure the resampling deals with NaN values correctly."""
        # Arrange
        scale_fine = 1.5
        scale_coarse = 1.5 * scaling_factor
        image = ImageContainer(
            data=image_rectangular_with_nans.data,
            metadata=MetaData(scale=Pair(scale_fine, scale_coarse)),
        )
        expected_shape = (
            int(round(image.height * scaling_factor)),
            int(round(image.width)),
        )

        # Act
        result = IsotropicResample(image)
        sample = unwrap_result(result)
        # Assert
        assert sample.metadata.scale == (scale_fine, scale_fine), (
            f"Scale should now be the minimum {scale_fine}"
        )
        assert (sample.height, sample.width) == expected_shape, (
            f"New width should be (original_width * (original_scale_x / target_scale)), but got: ({sample.height}, {sample.width})"
        )
