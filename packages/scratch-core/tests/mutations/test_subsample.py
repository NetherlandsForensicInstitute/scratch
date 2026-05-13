from math import ceil
from pathlib import Path

import numpy as np
import pytest
from container_models.scan_image import ScanImage
from mutations.spatial import Subsample


class TestSubSampleScanImage:
    # TODO: find a better test methology
    def test_subsample_matches_baseline_output(
        self, baseline_images_dir: Path, scan_image_replica: ScanImage
    ) -> None:
        # arrange
        verified = np.load(baseline_images_dir / "replica_subsampled.npy")
        # act
        result = Subsample(step_size_x=10, step_size_y=15)(
            scan_image_replica,
        )
        subsampled = result
        # assert
        assert np.allclose(
            subsampled.data,
            verified,
            equal_nan=True,
            atol=0.001,
        )

    @pytest.mark.parametrize(
        "step_size_x, step_size_y", [(1, 1), (10, 10), (25, 25), (25, 50)]
    )
    def test_subsample_matches_size(
        self, scan_image: ScanImage, step_size_x: int, step_size_y: int
    ):
        # Arrange
        expected_height = ceil(scan_image.data.shape[0] / step_size_y)
        expected_width = ceil(scan_image.data.shape[1] / step_size_x)

        # Act
        result = Subsample(step_size_x=step_size_x, step_size_y=step_size_y)(
            scan_image,
        )
        subsampled = result
        #  Assert
        assert subsampled.data.shape == (expected_height, expected_width)

    @pytest.mark.parametrize(
        ("step_x", "step_y"),
        [
            pytest.param(1, 1, id="default value"),
            pytest.param(10, 1, id="only x"),
            pytest.param(1, 10, id="only y"),
            pytest.param(10, 5, id="different x and y"),
        ],
    )
    def test_subsample_updates_scan_image_scales(
        self, scan_image: ScanImage, step_x: int, step_y: int
    ) -> None:
        # Act
        result = Subsample(step_size_x=step_x, step_size_y=step_y)(scan_image)
        subsampled = result
        # Assert
        assert np.isclose(subsampled.scale_x, scan_image.scale_x * step_x, atol=1.0e-3)
        assert np.isclose(subsampled.scale_y, scan_image.scale_y * step_y, atol=1.0e-3)

    @pytest.mark.parametrize(
        "step_size_x, step_size_y",
        [(-2, 2), (0, 0), (0, 3), (2, -1), (-1, -1), (1e3, 1e4)],
    )
    def test_subsample_rejects_incorrect_sizes(
        self, scan_image: ScanImage, step_size_x: int, step_size_y: int
    ):
        # Act
        with pytest.raises(ValueError):
            Subsample(step_size_x=step_size_x, step_size_y=step_size_y)(
                scan_image,
            )

    def test_subsample_skips_when_given_step_size_of_one(
        self, scan_image: ScanImage
    ) -> None:
        """
        Test when given the subsample the stepsize of one in both directions,
        it doesn't compute the whole image but just returns the original.
        """
        # Act
        result = Subsample(step_size_x=1, step_size_y=1)(scan_image)
        subsampled = result
        # Assert
        assert subsampled is scan_image, "Expected the same object to be returned"
