from collections.abc import Sequence
from functools import partial
from typing import Annotated, TypeAlias

from numpy import array, bool_, float64, number, uint8
from numpy.typing import DTypeLike, NDArray
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


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


ScanMapRGBA: TypeAlias = Annotated[
    NDArray[uint8],
    BeforeValidator(partial(coerce_to_array, uint8)),
    PlainSerializer(serialize_ndarray),
]

ScanMap2DArray = ScanVectorField2DArray = UnitVector3DArray = Annotated[
    NDArray[float64],
    BeforeValidator(partial(coerce_to_array, float64)),
    PlainSerializer(serialize_ndarray),
]

MaskArray = Annotated[
    NDArray[bool_],
    BeforeValidator(partial(coerce_to_array, bool_)),
    PlainSerializer(serialize_ndarray),
]


class ConfigBaseModel(BaseModel):
    """Base model with common configuration for all pydantic models in this project."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        regex_engine="rust-regex",
    )


class PointCloud(BaseModel, arbitrary_types_allowed=True):
    xs: NDArray
    ys: NDArray
    zs: NDArray
