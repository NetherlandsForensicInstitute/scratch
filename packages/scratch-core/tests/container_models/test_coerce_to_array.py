from functools import partial

import numpy as np
import pytest
from pydantic import ConfigDict, TypeAdapter, ValidationError

from container_models.base import FloatArray1D, FloatArray2D, UInt8Array3D

_TypeAdapter = partial(TypeAdapter, config=ConfigDict(arbitrary_types_allowed=True))


@pytest.mark.parametrize(
    "array_type, input_sequence",
    [
        pytest.param(FloatArray1D, [1.5, 2.5, 3.5], id="1d float list"),
        pytest.param(FloatArray1D, (1.0, 2.0, 3.0), id="1d float tuple"),
        pytest.param(FloatArray2D, [[1.0, 2.0], [3.0, 4.0]], id="2d float nested list"),
        pytest.param(UInt8Array3D, [[[10, 20, 30]]], id="3d uint8 list"),
        pytest.param(FloatArray1D, [], id="empty list"),
    ],
)
def test_coerce_sequence_to_array(array_type, input_sequence):
    """Test converting sequences (lists, tuples, nested lists) to arrays."""
    adapter = _TypeAdapter(array_type)
    result = adapter.validate_python(input_sequence)
    assert isinstance(result, np.ndarray)


def test_passthrough_existing_array():
    """Test that existing ndarray input is passed through without conversion."""
    adapter = _TypeAdapter(FloatArray1D)
    input_array = np.array([1.0, 2.0, 3.0])
    result = adapter.validate_python(input_array)
    assert np.array_equal(result, input_array)


@pytest.mark.parametrize(
    "input_value",
    [
        pytest.param([[[256]]], id="uint8 overflow"),
        pytest.param([[[-1]]], id="uint8 underflow"),
    ],
)
def test_coerce_out_of_range_list_raises_error(input_value):
    """Test that out-of-range list values raise ValueError."""
    adapter = _TypeAdapter(UInt8Array3D)
    with pytest.raises(ValidationError, match=r"Array's value\(s\) out of range"):
        adapter.validate_python(input_value)


@pytest.mark.parametrize(
    "input_value",
    [
        pytest.param(["not", "numbers"], id="list_of_strings"),
        pytest.param([1, "mixed", 3], id="mixed_types"),
    ],
)
def test_coerce_invalid_list_data_raises_error(input_value):
    """Test that non-numeric data raises ValueError."""
    adapter = _TypeAdapter(FloatArray1D)
    with pytest.raises(ValidationError):
        adapter.validate_python(input_value)
