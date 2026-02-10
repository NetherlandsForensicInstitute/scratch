from functools import partial

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter, ValidationError
from typing import Any

from container_models.base import (
    UInt8Array3D,
    FloatArray1D,
    FloatArray2D,
    FloatArray4D,
    BoolArray2D,
)

_TypeAdapter = partial(TypeAdapter, config=ConfigDict(arbitrary_types_allowed=True))


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
def test_array_valid_shape(array_type: Any, array: NDArray[Any]) -> None:
    """Test valid shape validation for various array types."""
    adapter = _TypeAdapter(array_type)
    result = adapter.validate_python(array)
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
    array_type: type,
    array: NDArray[Any],
    expected_match: str,
) -> None:
    """Test shape validation failures for various array types."""
    adapter = _TypeAdapter(array_type)
    with pytest.raises(ValidationError, match=expected_match):
        adapter.validate_python(array)
