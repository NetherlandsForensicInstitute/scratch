import numpy as np
import pytest

from container_models.base import coerce_to_array


@pytest.mark.parametrize(
    "dtype, input_sequence",
    [
        pytest.param(np.float64, [1.5, 2.5, 3.5], id="list float64"),
        pytest.param(np.uint8, [10, 20, 30], id="list uint8"),
        pytest.param(np.float32, (1.0, 2.0, 3.0), id="tuple float32"),
        pytest.param(np.float64, [[1.0, 2.0], [3.0, 4.0]], id="nested list"),
        pytest.param(np.uint8, [255], id="single value"),
        pytest.param(np.float64, [], id="empty"),
    ],
)
def test_coerce_sequence_to_array(dtype, input_sequence):
    """Test converting sequences (lists, tuples, nested lists) to arrays."""
    result = coerce_to_array(dtype, input_sequence)
    assert isinstance(result, np.ndarray)
    assert result.dtype == dtype
    assert np.array_equal(result, np.array(input_sequence, dtype))


@pytest.mark.parametrize(
    "dtype, input_value",
    [
        pytest.param(np.float64, np.array([1], dtype=np.int32), id="existing_array"),
        pytest.param(np.float64, None, id="None"),
    ],
)
def test_passthrough_behavior(dtype, input_value):
    """Test that certain inputs are returned as-is without conversion."""
    result = coerce_to_array(dtype, input_value)
    assert (
        result is input_value
        if input_value is None
        else np.array_equal(result, input_value)  # type: ignore
    )


@pytest.mark.parametrize(
    "dtype, input_value",
    [
        pytest.param(np.uint8, [256], id="uint8 overflow"),
        pytest.param(np.uint8, [-1], id="uint8 underflow"),
        pytest.param(np.int8, [128], id="int8 overflow"),
        pytest.param(np.int8, [-129], id="int8 underflow"),
    ],
)
def test_coerce_out_of_range_list_raises_error(dtype, input_value):
    """Test that out-of-range list values raise ValueError."""
    with pytest.raises(ValueError, match=r"Array's value\(s\) out of range"):
        coerce_to_array(dtype, input_value)


@pytest.mark.parametrize(
    "dtype, input_value",
    [
        pytest.param(np.float16, ["not", "numbers"], id="list_of_strings"),
        pytest.param(np.int32, [1, "mixed", 3], id="mixed_types"),
    ],
)
def test_coerce_invalid_list_data_raises_error(dtype, input_value):
    """Test that non-numeric data raises ValueError."""
    with pytest.raises(ValueError):
        coerce_to_array(dtype, input_value)
