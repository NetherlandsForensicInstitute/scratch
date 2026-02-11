"""Tests for image container classes."""

from pathlib import Path
import numpy as np
import pytest
from scipy.constants import micro

from container_models.base import DepthData, Pair
from container_models.image import MetaData, ImageContainer


class TestMetaData:
    """Tests for MetaData class."""

    @pytest.mark.parametrize(
        ("scale", "expected"),
        [
            pytest.param(Pair(1.0, 1.0), True, id="equal_scales"),
            pytest.param(Pair(2.5, 2.5), True, id="equal_float_scales"),
            pytest.param(Pair(1.0, 2.0), False, id="different_scales"),
            pytest.param(Pair(micro, micro), True, id="small_equal_scales"),
        ],
    )
    def test_is_isotropic(self, scale: Pair, expected: bool) -> None:
        # Arrange
        metadata = MetaData(scale=scale)
        # Act
        result = metadata.is_isotropic
        # Assert
        assert result is expected

    def test_central_diff_scales(self) -> None:
        # Arrange
        metadata = MetaData(scale=Pair(4.0, 6.0))
        # Act
        result = metadata.central_diff_scales
        # Assert
        assert result == Pair(2.0, 3.0)

    def test_central_diff_scales_with_micro(self) -> None:
        # Arrange
        metadata = MetaData(scale=Pair(4 * micro, 4 * micro))
        expected = Pair(2 * micro, 2 * micro)
        # Act
        result = metadata.central_diff_scales
        # Assert
        assert np.allclose(result, expected)


class TestImageContainer:
    """Tests for ImageContainer class."""

    @pytest.fixture
    def simple_image(self, flat_scale: MetaData) -> ImageContainer:
        """Create a simple 10x20 image container."""
        data = np.ones((10, 20), dtype=np.float64)
        return ImageContainer(data=data, metadata=flat_scale)

    @pytest.fixture
    def process_image(self, flat_scale: MetaData) -> ImageContainer:
        """Create a simple process image."""
        return ImageContainer(
            data=np.array([[100.0, 150.0], [200.0, 250.0]], dtype=np.float64),
            metadata=flat_scale,
        )

    @pytest.fixture
    def process_image_with_nan(self, flat_scale: MetaData) -> ImageContainer:
        """Create a process image with NaN values."""
        return ImageContainer(
            data=np.array([[100.0, np.nan], [np.nan, 250.0]], dtype=np.float64),
            metadata=flat_scale,
        )

    @pytest.fixture
    def mask_image(self, flat_scale: MetaData) -> ImageContainer:
        """Create a mask image."""
        return ImageContainer(
            data=np.array([[1, 0, 1], [0, 1, 0]]),
            metadata=flat_scale,
        )

    def test_shape(self, simple_image: ImageContainer) -> None:
        # Assert
        assert simple_image.height == simple_image.data.shape[0] == 10
        assert simple_image.width == simple_image.data.shape[1] == 20

    def test_center(self, simple_image: ImageContainer) -> None:
        # Arrange - via fixture
        # central_diff_scales = (0.5, 0.5)
        # center = ((0.5 - 1) * 20, (0.5 - 1) * 10) = (-10, -5)
        expected = Pair(-10.0, -5.0)
        # Act
        result = simple_image.center
        # Assert
        assert isinstance(result, Pair)
        assert result == expected

    @pytest.mark.parametrize(
        ("data1", "data2", "expected"),
        [
            pytest.param(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                True,
                id="equal_data",
            ),
            pytest.param(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[1.0, 2.0], [3.0, 5.0]]),
                False,
                id="different_data",
            ),
            pytest.param(
                np.array([[np.nan, 2.0], [3.0, 4.0]]),
                np.array([[np.nan, 2.0], [3.0, 4.0]]),
                True,
                id="equal_with_nan",
            ),
            pytest.param(
                np.array([[np.nan, 2.0]]),
                np.array([[1.0, 2.0]]),
                False,
                id="nan_vs_value",
            ),
        ],
    )
    def test_equality(
        self, data1: DepthData, data2: DepthData, expected: bool, flat_scale: MetaData
    ) -> None:
        # Arrange
        img1 = ImageContainer(data=data1, metadata=flat_scale)
        img2 = ImageContainer(data=data2, metadata=flat_scale)

        # Act
        result = img1 == img2

        # Assert
        assert result is expected

    def test_equality_with_non_image_container(self, flat_scale: MetaData) -> None:
        # Arrange
        data = np.ones((5, 5), dtype=np.float64)
        img = ImageContainer(data=data, metadata=flat_scale)

        # Act
        result_string = img == "not an image"
        result_int = img == 42

        # Assert - When __eq__ returns NotImplemented, Python falls back to identity
        assert result_string is False
        assert result_int is False

    # TODO: need better test
    def test_valid_mask_shape(self, mask_image: ImageContainer) -> None:
        # Act
        result = mask_image.valid_mask
        # Assert - Bool arrays don't have NaN, so all should be valid
        assert result.shape == mask_image.data.shape
        assert result.all()

    # TODO: need better test
    def test_valid_data_length(self, mask_image: ImageContainer) -> None:
        # Act
        result = mask_image.valid_data
        # Assert
        assert len(result) == 6  # All 6 values are valid

    def test_valid_mask_is_readonly(self, mask_image: ImageContainer) -> None:
        # Act
        result = mask_image.valid_mask
        # Assert
        assert not result.flags.writeable

    def test_valid_data_is_readonly(self, mask_image: ImageContainer) -> None:
        # Act
        result = mask_image.valid_data
        # Assert
        assert not result.flags.writeable

    def test_rgba_shape(self, process_image: ImageContainer) -> None:
        # Act
        result = process_image.rgba
        # Assert
        assert result.shape == (2, 2, 4)

    def test_rgba_alpha_channel_opaque(self, process_image: ImageContainer) -> None:
        # Act
        result = process_image.rgba
        # Assert - All pixels should be fully opaque (255) when no NaN
        assert np.all(result[..., 3] == 255)

    def test_rgba_alpha_channel_with_nan(
        self, process_image_with_nan: ImageContainer
    ) -> None:
        # Act
        result = process_image_with_nan.rgba
        # Assert - NaN pixels should have alpha=0, valid pixels alpha=255
        assert result[0, 0, 3] == 255  # valid
        assert result[0, 1, 3] == 0  # nan
        assert result[1, 0, 3] == 0  # nan
        assert result[1, 1, 3] == 255  # valid


@pytest.mark.parametrize(
    ("exporter", "filename"),
    [
        pytest.param("export_png", "test.png", id="png"),
        pytest.param("export_x3p", "test.x3p", id="x3p"),
    ],
)
class TestFileExport:
    def test_export_x3p_returns_failure_when_write_fails(
        self, exporter: str, filename: str, image_replica: ImageContainer
    ):
        """Test that save returns IOFailure when write operation fails."""
        # Arrange
        export = getattr(image_replica, exporter)

        # Act and Assert
        with pytest.raises(IOError, match="No such file or directory"):
            _ = export(output_path=Path(f"nonexistent_dir/{filename}"))

    def test_export_x3p_returns_success_on_valid_input(
        self,
        exporter: str,
        filename: str,
        image_replica: ImageContainer,
        tmp_path: Path,
    ):
        """Test that save_to_x3p returns IOSuccess(None) when save succeeds."""
        # Arrange
        output_path = tmp_path / filename
        export = getattr(image_replica, exporter)
        # Act
        result = export(output_path=output_path)
        # Assert
        assert result == output_path
        assert result.exists()
