from math import ceil
from pathlib import Path

import numpy as np
import pytest
from returns.pipeline import is_successful
from scipy.constants import micro
from surfalize import Surface
from unittest.mock import patch

from container_models.scan_image import ScanImage
from parsers import load_scan_image, subsample_scan_image
from parsers.loaders import make_isotropic

from ..helper_function import unwrap_result


@pytest.fixture(scope="class")
def filepath(scans_dir: Path, request: pytest.FixtureRequest):
    return scans_dir / request.param


@pytest.mark.parametrize(
    "filepath",
    [
        "Klein_non_replica_mode.al3d",
        "Klein_non_replica_mode_X3P_Scratch.x3p",
    ],
    indirect=True,
)
class TestLoadScanImage:
    def test_load_scan_data_matches_size(self, filepath: Path) -> None:
        # Arrange
        surface = Surface.load(filepath)
        # Act
        result = load_scan_image(filepath)
        scan_image = unwrap_result(result)

        # Assert
        assert scan_image.data.shape == (
            ceil(surface.data.shape[0]),
            ceil(surface.data.shape[1]),
        )
        assert scan_image.scale_y == surface.step_y * micro
        assert scan_image.scale_x == surface.step_x * micro


class TestLoadScanImageCaching:
    class FakeSurfaceOne:
        pass

    class FakeSurfaceTwo:
        pass

    @pytest.fixture(autouse=True)
    def empty_cache_for_test(self):
        load_scan_image.cache_clear()
        yield

    def test_load_scan_image_is_cached(self, tmp_path: Path) -> None:
        # Arrange
        scan_file = tmp_path / "scan.x3p"
        with patch(
            "parsers.loaders.Surface.load", return_value=self.FakeSurfaceOne()
        ) as mock_load:
            # Act
            image_1 = load_scan_image(scan_file)
            image_2 = load_scan_image(scan_file)

        # Assert
        assert image_1 is image_2, "same object expected due to caching"
        assert mock_load.call_count == 1, (
            "Surface.load should be called only once due to caching"
        )

    def test_load_scan_image_only_caches_one_image(self, tmp_path: Path) -> None:
        # Arrange
        scan_file_1 = tmp_path / "scan_1.x3p"
        scan_file_2 = tmp_path / "scan_2.x3p"

        with patch(
            "parsers.loaders.Surface.load",
            side_effect=[
                self.FakeSurfaceOne(),
                self.FakeSurfaceTwo(),
            ],
        ):
            # Act
            _image_1 = load_scan_image(scan_file_1)
            _image_2 = load_scan_image(scan_file_1)
            _image_3 = load_scan_image(scan_file_2)

        # Assert
        info = load_scan_image.cache_info()
        assert info.hits == 1, "one cache hit expected"
        assert info.misses == 2, "two different files loaded"
        assert info.currsize == 1, "Cache should only hold one item"


class TestSubSampleScanImage:
    # TODO: find a better test methology
    def test_subsample_matches_baseline_output(
        self, baseline_images_dir: Path, scan_image_replica: ScanImage
    ) -> None:
        # arrange
        verified = np.load(baseline_images_dir / "replica_subsampled.npy")
        # act
        result = subsample_scan_image(scan_image_replica, 10, 15)
        subsampled = unwrap_result(result)
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
        result = subsample_scan_image(scan_image, step_size_x, step_size_y)
        subsampled = unwrap_result(result)

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
        result = subsample_scan_image(scan_image, step_x, step_y)
        subsampled = unwrap_result(result)

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
        result = subsample_scan_image(scan_image, step_size_x, step_size_y)

        # Assert
        assert not is_successful(result)

    def test_subsample_skips_when_given_step_size_of_one(
        self, scan_image: ScanImage
    ) -> None:
        """
        Test when given the subsample the stepsize of one in both directions,
        it doesn't compute the whole image but just returns the original.
        """
        # Act
        result = subsample_scan_image(scan_image, 1, 1)
        subsampled = unwrap_result(result)

        # Assert
        assert subsampled is scan_image, "Expected the same object to be returned"

    def test_make_isotropic_no_op(self):
        """Ensure no resampling occurs if pixels are already square."""
        scan_image = ScanImage(scale_x=0.5, scale_y=0.5, data=np.zeros((100, 100)))

        result = unwrap_result(make_isotropic(scan_image))

        assert result.scale_x == 0.5
        assert result.scale_y == 0.5
        assert result.data.shape == (100, 100)
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
        expected_width, expected_height = 200, 100

        result = unwrap_result(make_isotropic(scan_image))

        assert result.scale_x == 1.0, "Scale should now be the minimum of the two (1.0)"
        assert result.scale_y == 1.0, "Scale should now be the minimum of the two (1.0)"
        assert result.data.shape == (expected_height, expected_width), (
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

        result = unwrap_result(make_isotropic(scan_image))

        assert result.meta_data == scan_image.meta_data
        assert np.min(result.data) == 0
        assert np.max(result.data) == 3000
        assert result.data.dtype == scan_image.data.dtype

    @pytest.mark.parametrize("scaling_factor", [4, 7.6, 8.1, 10.11])
    def test_make_isotropic_handles_nans(
        self, scan_image_rectangular_with_nans: ScanImage, scaling_factor: float
    ):
        """Ensure the resampling deals with NaN values correctly."""
        scan_image = ScanImage(
            data=scan_image_rectangular_with_nans.data,
            scale_x=1.5,
            scale_y=1.5 * scaling_factor,
        )

        result = unwrap_result(make_isotropic(scan_image))

        assert result.scale_x == 1.5
        assert result.scale_y == 1.5
        assert result.data.shape == (
            int(round(scan_image.height * scaling_factor)),
            int(round(scan_image.width)),
        )
        assert result.valid_mask.sum() / scan_image.valid_mask.sum() == pytest.approx(
            scaling_factor, abs=1e-3
        ), "The number of valid pixels / NaNs have not scaled correctly"
