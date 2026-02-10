from __future__ import annotations

from collections.abc import Callable, Sequence
from operator import add, mul, sub, truediv
from typing import Annotated, Iterable, NamedTuple

from numpy import array, bool_, float64, uint8
from numpy.typing import DTypeLike, NDArray
from pydantic import PlainSerializer, WrapValidator


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

    def __add__(self, other: Pair[T] | T) -> Pair[T]:  # type: ignore[override]
        return self._apply(add, other)

    def __radd__(self, other: T) -> Pair[T]:
        return self._rapply(add, other)

    def __sub__(self, other: Pair[T] | T) -> Pair[T]:
        return self._apply(sub, other)

    def __rsub__(self, other: T) -> Pair[T]:
        return self._rapply(sub, other)

    def __mul__(self, other: Pair[T] | T) -> Pair[T]:  # type: ignore[override]
        return self._apply(mul, other)

    def __rmul__(self, other: T) -> Pair[T]:  # type: ignore[override]
        return self._rapply(mul, other)

    def __truediv__(self, other: Pair[T] | T) -> Pair[T]:
        return self._apply(truediv, other)

    def __rtruediv__(self, other: T) -> Pair[T]:
        return self._rapply(truediv, other)


type Coordinate = Pair[float]
type Factor = Pair[float]
type Scale = Pair[float]


def _ndarray(dtype: DTypeLike, n_dims: int) -> WrapValidator:
    def _validate(value, handler):
        if isinstance(value, Sequence):
            try:
                value = array(value, dtype=dtype)
            except OverflowError as ofe:
                raise ValueError("Array's value(s) out of range") from ofe

        result = handler(value)

        if result.ndim != n_dims:
            raise ValueError(
                f"Array shape mismatch, expected {n_dims} dimension(s), but got {result.ndim}"
            )
        return result

    return WrapValidator(_validate)


_serialize = PlainSerializer(lambda array_: array_.tolist())

# Shape and data types
type FloatArray1D = Annotated[NDArray[float64], _ndarray(float64, 1), _serialize]
type FloatArray2D = Annotated[NDArray[float64], _ndarray(float64, 2), _serialize]
type FloatArray3D = Annotated[NDArray[float64], _ndarray(float64, 3), _serialize]
type FloatArray4D = Annotated[NDArray[float64], _ndarray(float64, 4), _serialize]
type BoolArray1D = Annotated[NDArray[bool_], _ndarray(bool_, 1), _serialize]
type BoolArray2D = Annotated[NDArray[bool_], _ndarray(bool_, 2), _serialize]
type UInt8Array3D = Annotated[NDArray[uint8], _ndarray(uint8, 3), _serialize]

# Semantic context
type ImageRGBA = UInt8Array3D  # Shape: (H, W, 4)
type UnitVector = FloatArray1D  # Shape: (3,)
type DepthData = FloatArray2D  # Shape: (H, W)
type BinaryMask = BoolArray2D  # Shape: (H, W)
type VectorField = FloatArray3D  # Shape (H, W, 3)
