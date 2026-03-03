import numpy as np
import pytest
from container_models.base import BinaryMask

from preprocessors.pipelines import parse_mask_pipeline


@pytest.mark.integration
class TestParseMaskPipeline:
    @pytest.fixture(scope="class")
    def mask_array(self) -> BinaryMask:
        """Fixture for a 2D mask array."""
        return np.array([[1, 0, 1], [0, 0, 1]], dtype=np.bool)

    def test_pipeline_can_parse_mask(self, mask_array: BinaryMask) -> None:
        """Test that the pipeline can parse a 2D mask from binary data."""
        # Arrange
        raw_data = mask_array.tobytes(order="C")
        shape = mask_array.shape
        # Act
        parsed_mask = parse_mask_pipeline(raw_data, shape)
        # Assert
        assert np.array_equal(parsed_mask, mask_array)

    def test_pipeline_can_parse_bitpacked_mask(self, mask_bitpacked: bytes, mask_original: BinaryMask) -> None:
        """Test that the pipeline can parse a 2D mask from binary data."""
        # Act
        parsed_mask = parse_mask_pipeline(mask_bitpacked, shape=mask_original.shape, is_bitpacked=True)
        # Assert
        assert parsed_mask.shape == mask_original.shape
        assert np.array_equal(parsed_mask, mask_original)

    def test_pipeline_raises_on_incorrect_shape(self, mask_array: BinaryMask) -> None:
        """Test that the pipeline will raise an error if the shape is incorrect."""
        raw_data = mask_array.tobytes(order="C")
        incorrect_shape = (100, 150)
        with pytest.raises(ValueError, match="cannot reshape array"):
            _ = parse_mask_pipeline(raw_data, incorrect_shape)
