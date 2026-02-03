from collections.abc import Sequence
from functools import partial
from typing import Annotated, TypeAlias

from numpy import array, bool_, floating, number, uint8
from numpy.typing import DTypeLike, NDArray
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
)


def serialize_ndarray[T: number](array_: NDArray[T]) -> list[T]:
    """Serialize numpy array to a Python list for JSON serialization."""
    return array_.tolist()


def coerce_to_array[T: number](
    dtype: DTypeLike, value: Sequence[T] | NDArray[T] | None
) -> NDArray[T] | None:
    """
    Coerce input to dtype numpy array.

    Handles JSON deserialization where Python creates int64 integers by default.
    """
    if isinstance(value, Sequence):
        try:
            return array(value, dtype=dtype)
        except OverflowError as ofe:
            raise ValueError("Array's value(s) out of range") from ofe

    return value


def validate_shape(n_dims: int, value: NDArray) -> NDArray:
    if (array_dims := len(value.shape)) != n_dims:
        raise ValueError(
            f"Array shape mismatch, expected {n_dims} dimension(s), but got {array_dims}"
        )
    return value


# Tier 1: Base types
UInt8Array: TypeAlias = Annotated[
    NDArray[uint8],
    BeforeValidator(partial(coerce_to_array, uint8)),
    PlainSerializer(serialize_ndarray),
]
FloatArray: TypeAlias = Annotated[
    NDArray[floating],
    BeforeValidator(partial(coerce_to_array, floating)),
    PlainSerializer(serialize_ndarray),
]
BoolArray: TypeAlias = Annotated[
    NDArray[bool_],
    BeforeValidator(partial(coerce_to_array, bool_)),
    PlainSerializer(serialize_ndarray),
]

# Tier 2: Shape and data types
UInt8Array3D: TypeAlias = Annotated[
    UInt8Array, AfterValidator(partial(validate_shape, 3))
]
FloatArray1D: TypeAlias = Annotated[
    FloatArray, AfterValidator(partial(validate_shape, 1))
]
FloatArray2D: TypeAlias = Annotated[
    FloatArray, AfterValidator(partial(validate_shape, 2))
]
FloatArray3D: TypeAlias = Annotated[
    FloatArray, AfterValidator(partial(validate_shape, 3))
]
FloatArray4D: TypeAlias = Annotated[
    FloatArray, AfterValidator(partial(validate_shape, 4))
]
BoolArray2D: TypeAlias = Annotated[
    BoolArray, AfterValidator(partial(validate_shape, 2))
]

# Tier 3: Semantic context
ImageRGB: TypeAlias = UInt8Array3D  # Shape: (H, W, 3)
ImageRGBA: TypeAlias = UInt8Array3D  # Shape: (H, W, 4)
UnitVector: TypeAlias = FloatArray1D  # Shape: (3,)
DepthData: TypeAlias = FloatArray2D  # Shape: (H, W)
BinaryMask: TypeAlias = BoolArray2D  # Shape: (H, W)
VectorField: TypeAlias = FloatArray3D  # Shape (H, W, 3)
StriationProfile: TypeAlias = FloatArray2D  # Shape (N, 1)


class ConfigBaseModel(BaseModel):
    """Base model with common configuration for all pydantic models in this project."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        regex_engine="rust-regex",
    )
