from functools import partial
import pytest
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter, ValidationError
from typing import Any

from container_models.base import (
    validate_shape,
    UInt8Array3D,
    FloatArray1D,
    FloatArray2D,
    FloatArray4D,
    BoolArray2D,
)

_TypeAdapter = partial(TypeAdapter, config=ConfigDict(arbitrary_types_allowed=True))


class TestValidateShape:
    """Tests for the validate_shape validator."""

    @pytest.mark.parametrize(
        "n_dims,array",
        [
            pytest.param(
                1,
                np.array([1, 2, 3]),
                id="1d_success",
            ),
            pytest.param(
                2,
                np.array([[1, 2], [3, 4]]),
                id="2d_success",
            ),
            pytest.param(
                3,
                np.ones((5, 10, 3)),
                id="3d_success",
            ),
            pytest.param(
                4,
                np.ones((2, 3, 4, 5)),
                id="4d_success",
            ),
        ],
    )
    def test_validate_shape_success(self, n_dims: int, array: NDArray[Any]) -> None:
        """Test successful shape validation for various dimensions."""
        # Arrange is handled by parametrize

        # Act
        result = validate_shape(n_dims, array)

        # Assert
        assert np.array_equal(result, array)

    @pytest.mark.parametrize(
        "n_dims,array,expected_match",
        [
            pytest.param(
                1,
                np.array([[1, 2], [3, 4]]),
                "expected 1 dimension\\(s\\), but got 2",
                id="1d_expected_2d_given",
            ),
            pytest.param(
                2,
                np.array([1, 2, 3]),
                "expected 2 dimension\\(s\\), but got 1",
                id="2d_expected_1d_given",
            ),
            pytest.param(
                3,
                np.array([[1, 2], [3, 4]]),
                "expected 3 dimension\\(s\\), but got 2",
                id="3d_expected_2d_given",
            ),
            pytest.param(
                1,
                np.array(42),
                "expected 1 dimension\\(s\\), but got 0",
                id="scalar_array",
            ),
        ],
    )
    def test_validate_shape_mismatch(
        self, n_dims: int, array: NDArray[Any], expected_match: str
    ) -> None:
        """Test shape validation failures with various dimension mismatches."""
        # Arrange is handled by parametrize

        # Act & Assert
        with pytest.raises(ValueError, match=expected_match):
            validate_shape(n_dims, array)


class TestPydanticShapeValidation:
    """Test shape validation at Tier 2 level using Pydantic models."""

    @pytest.mark.parametrize(
        "array_type,array",
        [
            pytest.param(
                FloatArray1D,
                np.array([1.0, 2.0, 3.0]),
                id="float_array_1d",
            ),
            pytest.param(
                FloatArray2D,
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                id="float_array_2d",
            ),
            pytest.param(
                UInt8Array3D,
                np.ones((10, 20, 4), dtype=np.uint8),
                id="uint8_array_3d",
            ),
            pytest.param(
                BoolArray2D,
                np.array([[True, False], [False, True]]),
                id="bool_array_2d",
            ),
            pytest.param(
                FloatArray4D,
                np.ones((2, 3, 4, 5)),
                id="float_array_4d",
            ),
        ],
    )
    def test_array_valid_shape(self, array_type: Any, array: NDArray[Any]) -> None:
        """Test valid shape validation for various array types."""
        # Arrange
        adapter = _TypeAdapter(array_type)

        # Act
        result = adapter.validate_python(array)

        # Assert
        assert np.array_equal(result, array)

    @pytest.mark.parametrize(
        "array_type,array,expected_match",
        [
            pytest.param(
                FloatArray1D,
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                "expected 1 dimension\\(s\\), but got 2",
                id="float_array_1d_invalid",
            ),
            pytest.param(
                FloatArray2D,
                np.array([1.0, 2.0, 3.0]),
                "expected 2 dimension\\(s\\), but got 1",
                id="float_array_2d_invalid",
            ),
            pytest.param(
                UInt8Array3D,
                np.ones((10, 20), dtype=np.uint8),
                "expected 3 dimension\\(s\\), but got 2",
                id="uint8_array_3d_invalid",
            ),
            pytest.param(
                BoolArray2D,
                np.array([True, False, True]),
                "expected 2 dimension\\(s\\), but got 1",
                id="bool_array_2d_invalid",
            ),
        ],
    )
    def test_array_invalid_shape(
        self,
        array_type: type,
        array: NDArray[Any],
        expected_match: str,
    ) -> None:
        """Test shape validation failures for various array types."""
        # Arrange
        adapter = _TypeAdapter(array_type)

        # Act & Assert
        with pytest.raises(ValidationError, match=expected_match):
            adapter.validate_python(array)
