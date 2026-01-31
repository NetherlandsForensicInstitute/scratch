from collections.abc import Sequence
from functools import partial
from typing import Annotated

from numpy import array, bool_, floating, number, uint8
from numpy.typing import DTypeLike, NDArray
from pydantic import AfterValidator, BeforeValidator


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


type BaseType[T: number | bool_] = Annotated[
    NDArray[T],
    BeforeValidator(partial(coerce_to_array, type(T))),
    AfterValidator(serialize_ndarray),
]

# Tier 1: Base types
type UInt8Array = BaseType[uint8]
type FloatArray = BaseType[floating]
type BoolArray = BaseType[bool_]

# Tier 2: Shape and data types
type UInt8Array3D = Annotated[UInt8Array, AfterValidator(partial(validate_shape, 3))]
type FloatArray1D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 1))]
type FloatArray2D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 2))]
type FloatArray3D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 3))]
type FloatArray4D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 4))]
type BoolArray2D = Annotated[BoolArray, AfterValidator(partial(validate_shape, 2))]

# Tier 3: Semantic context
type ImageRGBA = UInt8Array3D  # Shape: (H, W, 4)
type UnitVector = FloatArray1D  # Shape: (3,)
type DepthData = FloatArray2D  # Shape: (H, W)
type BinaryMask = BoolArray2D  # Shape: (H, W)
type VectorField = FloatArray3D  # Shape (H, W, 3)
