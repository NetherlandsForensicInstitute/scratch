from mutations.spatial import Resample
import numpy as np
import pytest

from container_models.scan_image import ScanImage


@pytest.fixture
def simple_scan_image() -> ScanImage:
    """Create a simple 10x10 scan image for testing."""
    return ScanImage(
        data=np.arange(100, dtype=np.float64).reshape(10, 10),
        scale_x=1,
        scale_y=1,
    )


@pytest.fixture
def scan_image_with_mask(simple_scan_image: ScanImage) -> ScanImage:
    """Create a scan image with a mask for testing."""
    mask = np.ones((10, 10), dtype=bool)
    mask[0:2, 0:2] = False
    return simple_scan_image.model_copy(update={"mask": mask})


class TestResampleScanImage:
    @pytest.mark.parametrize(
        "y_factor,x_factor,expected_shape",
        [
            pytest.param(
                2.0,
                2.0,
                (5, 5),
                id="downsample_by_2x",
            ),
            pytest.param(
                0.5,
                0.5,
                (20, 20),
                id="upsample_by_2x",
            ),
            pytest.param(
                1.0,
                1.0,
                (10, 10),
                id="no_scaling",
            ),
            pytest.param(
                2.0,
                1.0,
                (5, 10),
                id="downsample_x_only",
            ),
            pytest.param(
                1.0,
                2.0,
                (10, 5),
                id="downsample_y_only",
            ),
            pytest.param(3.67, 3.67, (2.7, 2.7), id="floats are also fine"),
        ],
    )
    def test_resampling_changes_shape(
        self,
        simple_scan_image: ScanImage,
        y_factor: float,
        x_factor: float,
        expected_shape: tuple[int, int],
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        expected_resampled_output_shape = (
            1 / y_factor * simple_scan_image.height,
            1 / x_factor * simple_scan_image.width,
        )
        resampling = Resample(target_shape=expected_resampled_output_shape)
        # Act
        result = resampling(simple_scan_image).unwrap()
        # Assert
        assert result.data.shape[0] == round(expected_shape[0], 0)
        assert result.data.shape[1] == round(expected_shape[1], 0)
        assert (
            f"Resampling image array to new size: {round(float(expected_shape[0]), 1)}/{round(float(expected_shape[1]), 1)} with scale: x:{round(x_factor, 1)}, y:{round(y_factor, 1)}"
            in caplog.messages
        )

    @pytest.mark.parametrize(
        ("y_factor", "x_factor"),
        [
            pytest.param(2.0, 2.0, id="downsample"),
            pytest.param(0.5, 0.5, id="upsample"),
            pytest.param(2.0, 0.5, id="mixed_scaling"),
        ],
    )
    def test_resampling_updates_scale(
        self,
        simple_scan_image: ScanImage,
        y_factor: float,
        x_factor: float,
    ):
        # Arrange
        expected_resampled_output_shape = (
            1 / y_factor * simple_scan_image.height,
            1 / x_factor * simple_scan_image.width,
        )
        original_scale_x = simple_scan_image.scale_x
        original_scale_y = simple_scan_image.scale_y
        resampling = Resample(target_shape=expected_resampled_output_shape)

        # Act
        result = resampling(simple_scan_image).unwrap()

        # Assert
        assert result.scale_x == original_scale_x * x_factor
        assert result.scale_y == original_scale_y * y_factor

    def test_resampling_preserves_data_properties(self, simple_scan_image: ScanImage):
        # Arrange
        factor = 2.0
        expected_resampled_output_shape = (
            1 / factor * simple_scan_image.height,
            1 / factor * simple_scan_image.width,
        )
        original_dtype = simple_scan_image.data.dtype
        original_ndim = simple_scan_image.data.ndim
        resampling = Resample(target_shape=expected_resampled_output_shape)

        # Act
        result = resampling(simple_scan_image).unwrap()

        # Assert
        assert result.data.dtype == original_dtype
        assert result.data.ndim == original_ndim
        assert result.data.ndim == 2

    def test_resampling_with_nan_values(self):
        # Arrange
        data = np.full((10, 10), np.nan, dtype=np.float64)
        data[5, 5] = 100.0
        scan_image = ScanImage(
            data=data,
            scale_x=1,
            scale_y=1,
        )
        factor = 2.0
        expected_resampled_output_shape = (
            1 / factor * data.shape[0],
            1 / factor * data.shape[1],
        )
        resampling = Resample(target_shape=expected_resampled_output_shape)
        # Act
        result = resampling(scan_image).unwrap()

        # Assert
        assert result.data.shape == (5, 5)
        assert np.isnan(result.data).any()
