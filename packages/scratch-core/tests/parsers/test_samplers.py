import numpy as np
import pytest

from container_models.base import Point
from container_models.scan_image import ScanImage
from parsers.samplers import resample_scan_image
from scipy.constants import micro


@pytest.fixture
def simple_scan_image() -> ScanImage:
    """Create a simple 10x10 scan image for testing."""
    return ScanImage(
        data=np.arange(100, dtype=np.float64).reshape(10, 10),
        scale_x=1 * micro,
        scale_y=1 * micro,
    )


@pytest.fixture
def scan_image_with_mask(simple_scan_image: ScanImage) -> ScanImage:
    """Create a scan image with a mask for testing."""
    mask = np.ones((10, 10), dtype=bool)
    mask[0:2, 0:2] = False
    return simple_scan_image.model_copy(update={"mask": mask})


class TestResampleScanImage:
    @pytest.mark.parametrize(
        "factors,expected_shape",
        [
            pytest.param(
                Point[float](2.0, 2.0),
                (5, 5),
                id="downsample_by_2x",
            ),
            pytest.param(
                Point[float](0.5, 0.5),
                (20, 20),
                id="upsample_by_2x",
            ),
            pytest.param(
                Point[float](1.0, 1.0),
                (10, 10),
                id="no_scaling",
            ),
            pytest.param(
                Point[float](2.0, 1.0),
                (10, 5),
                id="downsample_x_only",
            ),
            pytest.param(
                Point[float](1.0, 2.0),
                (5, 10),
                id="downsample_y_only",
            ),
        ],
    )
    def test_resampling_changes_shape(
        self,
        simple_scan_image: ScanImage,
        factors: Point[float],
        expected_shape: tuple[int, int],
    ):
        # Act
        result = resample_scan_image(simple_scan_image, factors)

        # Assert
        assert result.data.shape == expected_shape

    @pytest.mark.parametrize(
        "factors",
        [
            pytest.param(Point[float](2.0, 2.0), id="downsample"),
            pytest.param(Point[float](0.5, 0.5), id="upsample"),
            pytest.param(Point[float](2.0, 0.5), id="mixed_scaling"),
        ],
    )
    def test_resampling_updates_scale(
        self,
        simple_scan_image: ScanImage,
        factors: Point[float],
    ):
        # Arrange
        original_scale_x = simple_scan_image.scale_x
        original_scale_y = simple_scan_image.scale_y

        # Act
        result = resample_scan_image(simple_scan_image, factors)

        # Assert
        assert result.scale_x == original_scale_x * factors.x
        assert result.scale_y == original_scale_y * factors.y

    def test_resampling_without_mask_returns_no_mask(
        self, simple_scan_image: ScanImage
    ):
        # Arrange
        factors = Point[float](2.0, 2.0)

        # Act
        result = resample_scan_image(simple_scan_image, factors)

        # Assert
        assert result.mask is None

    def test_resampling_with_mask_resamples_mask(self, scan_image_with_mask: ScanImage):
        # Arrange
        factors = Point[float](2.0, 2.0)

        # Act
        result = resample_scan_image(scan_image_with_mask, factors)

        # Assert
        assert result.mask is not None
        assert result.mask.shape == result.data.shape
        assert result.mask.shape == (5, 5)

    def test_resampling_preserves_data_properties(self, simple_scan_image: ScanImage):
        # Arrange
        factors = Point[float](2.0, 2.0)
        original_dtype = simple_scan_image.data.dtype
        original_ndim = simple_scan_image.data.ndim

        # Act
        result = resample_scan_image(simple_scan_image, factors)

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
            scale_x=1e-6,
            scale_y=1e-6,
        )
        factors = Point[float](2.0, 2.0)

        # Act
        result = resample_scan_image(scan_image, factors)

        # Assert
        assert result.data.shape == (5, 5)
        assert np.isnan(result.data).any()
