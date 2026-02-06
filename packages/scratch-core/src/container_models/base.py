from __future__ import annotations
from collections.abc import Callable, Sequence
from functools import partial
from operator import add, mul, sub, truediv
from typing import Annotated, Iterable, NamedTuple

from numpy import array, bool_, floating, number, uint8
from numpy.typing import DTypeLike, NDArray
from pydantic import AfterValidator, BeforeValidator, PlainSerializer


class Pair[T](NamedTuple):
    x: T
    y: T

    def map(self, func: Callable, *, other: Iterable[T] | None = None) -> Pair[T]:
        if other:
            return Pair(*tuple(map(func, self, other)))
        return Pair(*tuple(map(func, self)))

    def _apply(self, op: Callable, other: Pair[T] | T) -> Pair[T]:
        if isinstance(other, Pair):
            return Pair(*map(op, self, other))
        return Pair(op(self.x, other), op(self.y, other))

    def _rapply(self, op: Callable, other: T) -> Pair[T]:
        return Pair(op(other, self.x), op(other, self.y))

    def __add__(self, other: Pair[T] | T) -> Pair[T]:
        return self._apply(add, other)

    def __radd__(self, other: T) -> Pair[T]:
        return self._rapply(add, other)

    def __sub__(self, other: Pair[T] | T) -> Pair[T]:
        return self._apply(sub, other)

    def __rsub__(self, other: T) -> Pair[T]:
        return self._rapply(sub, other)

    def __mul__(self, other: Pair[T] | T) -> Pair[T]:
        return self._apply(mul, other)

    def __rmul__(self, other: T) -> Pair[T]:
        return self._rapply(mul, other)

    def __truediv__(self, other: Pair[T] | T) -> Pair[T]:
        return self._apply(truediv, other)

    def __rtruediv__(self, other: T) -> Pair[T]:
        return self._rapply(truediv, other)


type Coordinate = Pair[float]
type Factor = Pair[float]
type Scale = Pair[float]


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
    PlainSerializer(serialize_ndarray),
]

# Tier 1: Base types
type FloatArray = BaseType[floating]
type BoolArray = BaseType[bool_]

# Tier 2: Shape and data types
type FloatArray1D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 1))]
type FloatArray2D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 2))]
type FloatArray3D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 3))]
type FloatArray4D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 4))]
type BoolArray1D = Annotated[BoolArray, AfterValidator(partial(validate_shape, 1))]
type BoolArray2D = Annotated[BoolArray, AfterValidator(partial(validate_shape, 2))]
type UInt8Array3D = Annotated[
    BaseType[uint8], AfterValidator(partial(validate_shape, 3))
]

# Tier 3: Semantic context
type ImageRGBA = UInt8Array3D  # Shape: (H, W, 4)
type UnitVector = FloatArray1D  # Shape: (3,)
type DepthData = FloatArray2D  # Shape: (H, W)
type BinaryMask = BoolArray2D  # Shape: (H, W)
type VectorField = FloatArray3D  # Shape (H, W, 3)
