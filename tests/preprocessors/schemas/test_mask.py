import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from preprocessors.schemas import Mask, MaskEncoder


class TestMaskEncoder:
    """Tests for MaskEncoder class."""

    def test_encode_should_convert_bytes_to_bool_bytes(self) -> None:
        """Test that encode converts any bytes to strict bool bytes (0 or 1)."""
        # Arrange
        input_bytes = b"\x04\x00\x00\x02"

        # Act
        result = MaskEncoder.encode(input_bytes)

        # Assert
        assert result == b"\x01\x00\x00\x01"

    def test_decode_should_accept_valid_bool_bytes(self) -> None:
        """Test that decode accepts bytes with only 0 and 1 values."""
        # Arrange
        valid_bytes = bytes([1, 0, 1, 0])

        # Act
        result = MaskEncoder.decode(valid_bytes)

        # Assert
        assert result == valid_bytes

    def test_decode_should_reject_empty_bytes(self) -> None:
        """Test that decode rejects empty bytes."""
        # Act & Assert
        with pytest.raises(ValueError, match="Cannot decode empty bytes"):
            MaskEncoder.decode(b"")

    @given(invalid_byte=st.integers(min_value=2, max_value=255))
    def test_decode_should_reject_non_bool_bytes(self, invalid_byte: int) -> None:
        """Test that decode rejects bytes with values other than 0 or 1."""
        # Arrange
        invalid_bytes = bytes([1, 0, invalid_byte, 1])

        # Act & Assert
        with pytest.raises(ValueError, match="Corrupted encoding"):
            MaskEncoder.decode(invalid_bytes)

    def test_get_json_format_should_return_mask_bytes(self) -> None:
        """Test that get_json_format returns the expected format string."""
        # Act
        result = MaskEncoder.get_json_format()

        # Assert
        assert result == "mask-bytes"

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encode followed by decode preserves data."""
        # Arrange
        original_bytes = b"\x01\x00\x00\x01"

        # Act
        encoded = MaskEncoder.encode(original_bytes)
        decoded = MaskEncoder.decode(encoded)

        # Assert
        assert decoded == encoded == original_bytes


class TestMask:
    """Tests for Mask model."""

    def test_should_create_mask_with_valid_data_and_shape(self) -> None:
        """Test that Mask can be created with valid data and shape."""
        # Arrange
        data = b"\x01\x00\x00\x01"
        shape = (2, 2)

        # Act
        mask = Mask(data=data, shape=shape)

        # Assert
        assert mask.data == data
        assert mask.shape == shape

    def test_mask_array_should_return_correct_numpy_array(self) -> None:
        """Test that mask_array property returns correct numpy array."""
        # Arrange
        shape = (2, 3)
        mask = Mask(data=b"\x01\x00\x01\x00\x01\x00", shape=shape)

        # Act
        result = mask.mask_array

        # Assert
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[True, False, True], [False, True, False]], dtype=np.bool_))

    def test_mask_array_should_be_cached(self) -> None:
        """Test that mask_array uses cached_property and returns same object."""
        # Arrange
        mask = Mask(data=b"\x01\x00\x00\x01", shape=(2, 2))

        # Act
        result1 = mask.mask_array
        result2 = mask.mask_array

        # Assert - cached_property returns the same object
        assert result1 is result2

    def test_should_reject_mask_with_invalid_shape(self) -> None:
        """Test that validation fails when data cannot be reshaped to given shape."""
        # Arrange
        shape = (2, 3)

        # Act & Assert
        with pytest.raises(ValidationError, match="Failed to decode mask data"):
            Mask(data=b"\x01\x00\x00\x01", shape=shape)

    @given(
        height=st.integers(min_value=1, max_value=50),
        width=st.integers(min_value=1, max_value=50),
        data=st.data(),
    )
    def test_mask_with_random_dimensions(self, height: int, width: int, data: st.DataObject) -> None:
        """Test Mask creation with various dimensions using hypothesis."""
        # Arrange
        arr = np.array(
            [[data.draw(st.booleans()) for _ in range(width)] for _ in range(height)],
            dtype=np.bool_,
        )
        mask_data = arr.tobytes()
        shape = (height, width)

        # Act
        mask = Mask(data=mask_data, shape=shape)

        # Assert
        assert mask.shape == shape
        np.testing.assert_array_equal(mask.mask_array, arr)

    def test_mask_serialization_to_dict(self) -> None:
        """Test that Mask can be serialized to dict."""
        # Arrange
        data = b"\x01\x00\x00\x01"
        shape = (2, 2)
        mask = Mask(data=data, shape=shape)

        # Act
        result = mask.model_dump()

        # Assert
        assert result.get("data") == data
        assert result.get("shape") == shape

    @pytest.mark.parametrize(
        ("data_bytes", "shape", "error_pattern"),
        [
            pytest.param(bytes([1, 0, 1, 0]), (4,), "Field required", id="1D array"),
            pytest.param(bytes([1, 0, 1, 0, 1, 0, 1, 0]), (2, 2, 2), "at most 2 items", id="3D array"),
            pytest.param(bytes([1, 0, 1, 0]), (2, 3), "Failed to decode mask data", id="shape mismatch"),
            pytest.param(b"", (0, 0), "Cannot decode empty bytes", id="empty bytes"),
        ],
    )
    def test_should_reject_invalid_mask_dimensions(
        self, data_bytes: bytes, shape: tuple[int, ...], error_pattern: str
    ) -> None:
        """Test that mask validation rejects non-2D arrays and invalid data."""
        # Act & Assert
        with pytest.raises(ValidationError, match=error_pattern):
            Mask(data=data_bytes, shape=shape)  # type: ignore
