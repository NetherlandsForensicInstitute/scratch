from container_models.base import Coordinate, Factor, Pair
from mutations.spatial import Resample
import numpy as np
import pytest

from container_models.image import ImageContainer, MetaData


@pytest.fixture
def simple_image(flat_scale: MetaData) -> ImageContainer:
    """Create a simple 10x10 scan image for testing."""
    return ImageContainer(
        data=np.arange(100, dtype=np.float64).reshape(10, 10), metadata=flat_scale
    )


@pytest.fixture
def image_with_mask(simple_image: ImageContainer) -> ImageContainer:
    """Create a scan image with a mask for testing."""
    mask = np.ones((10, 10), dtype=bool)
    mask[0:2, 0:2] = False
    return simple_image.model_copy(update={"mask": mask})


class TestResampleImageContainer:
    @pytest.mark.parametrize(
        "factors,expected_shape",
        [
            pytest.param(Pair(2.0, 2.0), (5, 5), id="downsample_by_2x"),
            pytest.param(Pair(0.5, 0.5), (20, 20), id="upsample_by_2x"),
            pytest.param(Pair(1.0, 1.0), (10, 10), id="no_scaling"),
            pytest.param(Pair(2.0, 1.0), (5, 10), id="downsample_x_only"),
            pytest.param(Pair(1.0, 2.0), (10, 5), id="downsample_y_only"),
        ],
    )
    def test_resampling_changes_shape(
        self,
        simple_image: ImageContainer,
        factors: Factor,
        expected_shape: tuple[int, int],
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        resampling = Resample(factors)
        # Act
        result = resampling(simple_image).unwrap()
        # Assert
        assert result.data.shape == expected_shape
        assert (
            f"Resampling image array to new size: {expected_shape[0]:.1f}/{expected_shape[1]:.1f}"
            in caplog.messages
        )

    def test_resampling_works_with_float(
        self, simple_image: ImageContainer, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        resampling = Resample(Pair(3.67, 3.67))
        # Act
        result = resampling(simple_image).unwrap()
        # Assert
        assert result.data.shape == (3, 3)
        assert "Resampling image array to new size: 2.7/2.7" in caplog.messages

    @pytest.mark.parametrize(
        "factors",
        [
            pytest.param(Pair(2.0, 2.0), id="downsample"),
            pytest.param(Pair(0.5, 0.5), id="upsample"),
            pytest.param(Pair(2.0, 0.5), id="mixed_scaling"),
        ],
    )
    def test_resampling_updates_scale(
        self, simple_image: ImageContainer, factors: Coordinate
    ):
        # Arrange
        original_scale = simple_image.metadata.scale
        resampling = Resample(factors)

        # Act
        result = resampling(simple_image).unwrap()

        # Assert
        assert result.metadata.scale.x == original_scale.x * factors.x
        assert result.metadata.scale.y == original_scale.y * factors.y

    def test_resampling_preserves_data_properties(self, simple_image: ImageContainer):
        # Arrange
        original_dtype = simple_image.data.dtype
        original_ndim = simple_image.data.ndim
        resampling = Resample(Pair(2.0, 2.0))

        # Act
        result = resampling(simple_image).unwrap()

        # Assert
        assert result.data.dtype == original_dtype
        assert result.data.ndim == original_ndim

    def test_resampling_with_nan_values(self, flat_scale: MetaData):
        # Arrange
        data = np.full((10, 10), np.nan, dtype=np.float64)
        data[5, 5] = 100.0
        image = ImageContainer(data=data, metadata=flat_scale)
        resampling = Resample(Pair(2.0, 2.0))
        # Act
        result = resampling(image).unwrap()

        # Assert
        assert result.data.shape == (5, 5)
        assert np.isnan(result.data).any()
