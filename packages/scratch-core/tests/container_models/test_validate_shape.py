import pytest
import numpy as np
from pydantic import ValidationError

from container_models.base import (
    validate_shape,
    UInt8Array3D,
    FloatArray1D,
    FloatArray2D,
    FloatArray4D,
    Int64Array2D,
    BoolArray2D,
    ConfigBaseModel,
)


class TestValidateShape:
    """Tests for the validate_shape validator."""

    def test_validate_shape_1d_success(self):
        arr = np.array([1, 2, 3])
        result = validate_shape(1, arr)
        assert np.array_equal(result, arr)

    def test_validate_shape_2d_success(self):
        arr = np.array([[1, 2], [3, 4]])
        result = validate_shape(2, arr)
        assert np.array_equal(result, arr)

    def test_validate_shape_3d_success(self):
        arr = np.ones((5, 10, 3))
        result = validate_shape(3, arr)
        assert np.array_equal(result, arr)

    def test_validate_shape_4d_success(self):
        arr = np.ones((2, 3, 4, 5))
        result = validate_shape(4, arr)
        assert np.array_equal(result, arr)

    def test_validate_shape_mismatch_1d_expected_2d_given(self):
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="expected 1 dimension\(s\), but got 2"):
            validate_shape(1, arr)

    def test_validate_shape_mismatch_2d_expected_1d_given(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="expected 2 dimension\(s\), but got 1"):
            validate_shape(2, arr)

    def test_validate_shape_mismatch_3d_expected_2d_given(self):
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="expected 3 dimension\(s\), but got 2"):
            validate_shape(3, arr)

    def test_validate_shape_scalar_array(self):
        """Test with 0-dimensional array (scalar)."""
        arr = np.array(42)
        with pytest.raises(ValueError, match="expected 1 dimension\(s\), but got 0"):
            validate_shape(1, arr)


class TestPydanticShapeValidation:
    """Test shape validation at Tier 2 level using Pydantic models."""

    def test_float_array_1d_valid(self):
        class Model(ConfigBaseModel):
            data: FloatArray1D

        arr = np.array([1.0, 2.0, 3.0])
        model = Model(data=arr)
        assert np.array_equal(model.data, arr)

    def test_float_array_1d_invalid_shape(self):
        class Model(ConfigBaseModel):
            data: FloatArray1D

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(
            ValidationError, match="expected 1 dimension\(s\), but got 2"
        ):
            Model(data=arr)

    def test_float_array_2d_valid(self):
        class Model(ConfigBaseModel):
            data: FloatArray2D

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = Model(data=arr)
        assert np.array_equal(model.data, arr)

    def test_float_array_2d_invalid_shape(self):
        class Model(ConfigBaseModel):
            data: FloatArray2D

        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(
            ValidationError, match="expected 2 dimension\(s\), but got 1"
        ):
            Model(data=arr)

    def test_uint8_array_3d_valid(self):
        class Model(ConfigBaseModel):
            data: UInt8Array3D

        arr = np.ones((10, 20, 4), dtype=np.uint8)
        model = Model(data=arr)
        assert np.array_equal(model.data, arr)

    def test_uint8_array_3d_invalid_shape(self):
        class Model(ConfigBaseModel):
            data: UInt8Array3D

        arr = np.ones((10, 20), dtype=np.uint8)
        with pytest.raises(
            ValidationError, match="expected 3 dimension\(s\), but got 2"
        ):
            Model(data=arr)

    def test_bool_array_2d_valid(self):
        class Model(ConfigBaseModel):
            data: BoolArray2D

        arr = np.array([[True, False], [False, True]])
        model = Model(data=arr)
        assert np.array_equal(model.data, arr)

    def test_bool_array_2d_invalid_shape(self):
        class Model(ConfigBaseModel):
            data: BoolArray2D

        arr = np.array([True, False, True])
        with pytest.raises(
            ValidationError, match="expected 2 dimension\(s\), but got 1"
        ):
            Model(data=arr)

    def test_int64_array_2d_valid(self):
        class Model(ConfigBaseModel):
            data: Int64Array2D

        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        model = Model(data=arr)
        assert np.array_equal(model.data, arr)

    def test_float_array_4d_valid(self):
        class Model(ConfigBaseModel):
            data: FloatArray4D

        arr = np.ones((2, 3, 4, 5))
        model = Model(data=arr)
        assert np.array_equal(model.data, arr)
