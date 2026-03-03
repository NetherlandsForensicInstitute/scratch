from collections.abc import Sequence
from functools import cached_property, partial
from typing import Annotated

from numpy import array, bool_, float64, floating, number, uint8
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


def coerce_to_array[T: number](dtype: DTypeLike, value: Sequence[T] | NDArray[T] | None) -> NDArray[T] | None:
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
    """Validate that an array has the expected number of dimensions."""
    if (array_dims := len(value.shape)) != n_dims:
        raise ValueError(f"Array shape mismatch, expected {n_dims} dimension(s), but got {array_dims}")
    return value


# Tier 1: Base types
type UInt8Array = Annotated[
    NDArray[uint8],
    BeforeValidator(partial(coerce_to_array, uint8)),
    PlainSerializer(serialize_ndarray),
]
type FloatArray = Annotated[
    NDArray[floating],
    BeforeValidator(partial(coerce_to_array, float64)),
    PlainSerializer(serialize_ndarray),
]
type BoolArray = Annotated[
    NDArray[bool_],
    BeforeValidator(partial(coerce_to_array, bool_)),
    PlainSerializer(serialize_ndarray),
]

# Tier 2: Shape and data types
type UInt8Array3D = Annotated[UInt8Array, AfterValidator(partial(validate_shape, 3))]
type FloatArray1D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 1))]
type FloatArray2D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 2))]
type FloatArray3D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 3))]
type FloatArray4D = Annotated[FloatArray, AfterValidator(partial(validate_shape, 4))]
type BoolArray2D = Annotated[BoolArray, AfterValidator(partial(validate_shape, 2))]

# Tier 3: Semantic context
type ImageRGB = UInt8Array3D  # Shape: (H, W, 3)
type ImageRGBA = UInt8Array3D  # Shape: (H, W, 4)
type UnitVector = FloatArray1D  # Shape: (3,)
type DepthData = FloatArray2D  # Shape: (H, W)
type BinaryMask = BoolArray2D  # Shape: (H, W)
type VectorField = FloatArray3D  # Shape (H, W, 3)
type StriationProfile = FloatArray1D  # Shape (N,)


class ConfigBaseModel(BaseModel):
    """Base model with common configuration for all pydantic models in this project."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        regex_engine="rust-regex",
        revalidate_instances="always",
    )

    def model_copy(self, *, update=None, deep=False):
        """Copy the model, revalidating cached properties when fields change."""
        copy = super().model_copy(update=update, deep=deep)
        if update:
            # Invalidate cached properties when any field changes
            self._clear_cached_properties(copy)
            # Validate model after updating
            copy = self.model_validate(copy, by_alias=True, by_name=True)
        return copy

    @staticmethod
    def _clear_cached_properties(instance: BaseModel):
        """Dynamically find and clear all cached_property values from instance."""
        for name in dir(type(instance)):
            attr = getattr(type(instance), name, None)
            if isinstance(attr, cached_property):
                instance.__dict__.pop(name, None)
