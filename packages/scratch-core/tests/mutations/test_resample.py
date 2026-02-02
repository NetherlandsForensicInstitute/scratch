from mutations.spatial import Resample
import numpy as np
import pytest

from container_models.base import Factors
from container_models.scan_image import ScanImage
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
                Factors[float](2.0, 2.0),
                (5, 5),
                id="downsample_by_2x",
            ),
            pytest.param(
                Factors[float](0.5, 0.5),
                (20, 20),
                id="upsample_by_2x",
            ),
            pytest.param(
                Factors[float](1.0, 1.0),
                (10, 10),
                id="no_scaling",
            ),
            pytest.param(
                Factors[float](2.0, 1.0),
                (10, 5),
                id="downsample_x_only",
            ),
            pytest.param(
                Factors[float](1.0, 2.0),
                (5, 10),
                id="downsample_y_only",
            ),
        ],
    )
    def test_resampling_changes_shape(
        self,
        simple_scan_image: ScanImage,
        factors: Factors[float],
        expected_shape: tuple[int, int],
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        resampling = Resample(factors=factors)
        # Act
        result = resampling(simple_scan_image).unwrap()
        # Assert
        assert result.data.shape == expected_shape
        assert (
            f"Resampling image array to new size: {float(expected_shape[0])}/{float(expected_shape[1])}"
            in caplog.messages
        )

    @pytest.mark.parametrize(
        "factors",
        [
            pytest.param(Factors[float](2.0, 2.0), id="downsample"),
            pytest.param(Factors[float](0.5, 0.5), id="upsample"),
            pytest.param(Factors[float](2.0, 0.5), id="mixed_scaling"),
        ],
    )
    def test_resampling_updates_scale(
        self,
        simple_scan_image: ScanImage,
        factors: Factors[float],
    ):
        # Arrange
        original_scale_x = simple_scan_image.scale_x
        original_scale_y = simple_scan_image.scale_y
        resampling = Resample(factors=factors)

        # Act
        result = resampling(simple_scan_image).unwrap()

        # Assert
        assert result.scale_x == original_scale_x * factors.x
        assert result.scale_y == original_scale_y * factors.y

    def test_resampling_preserves_data_properties(self, simple_scan_image: ScanImage):
        # Arrange
        factors = Factors[float](2.0, 2.0)
        original_dtype = simple_scan_image.data.dtype
        original_ndim = simple_scan_image.data.ndim
        resampling = Resample(factors=factors)

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
            scale_x=1e-6,
            scale_y=1e-6,
        )
        factors = Factors[float](2.0, 2.0)
        resampling = Resample(factors=factors)
        # Act
        result = resampling(scan_image).unwrap()

        # Assert
        assert result.data.shape == (5, 5)
        assert np.isnan(result.data).any()
