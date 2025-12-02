from functools import partial
from pydantic import BeforeValidator
from collections.abc import Sequence
from numpydantic import NDArray, Shape
from numpy import float64, uint8, array
from typing import Annotated
from numpy.typing import DTypeLike

HeightWidthShape = "*, *"
HeightWidthNLayers = f"{HeightWidthShape}, *"
HeightWidth3Layers = f"{HeightWidthShape}, 3"
UnitVector = "3"


def coerce_to_array(
    dtype: DTypeLike, value: Sequence | NDArray | None
) -> NDArray | None:
    """Coerce input to dtype numpy array.

    Handles JSON deserialization where Python creates int64 integers by default.
    """
    if isinstance(value, Sequence):
        try:
            return array(value, dtype=dtype)
        except OverflowError as ofe:
            raise ValueError("Array's value(s) out of range") from ofe

    return value


coerce_to_float64_array = partial(coerce_to_array, float64)

ScanMap2DArray = Annotated[
    NDArray[Shape[HeightWidthShape], float64],
    BeforeValidator(coerce_to_float64_array),
]  # type: ignore
MaskMap2DArray = Annotated[
    NDArray[Shape[HeightWidthShape], uint8],
    BeforeValidator(lambda x: coerce_to_array(uint8, x)),
]  # type: ignore
ScanTensor3DArray = Annotated[
    NDArray[Shape[HeightWidthNLayers], float64],
    BeforeValidator(coerce_to_float64_array),
]  # type: ignore
ScanVectorField2DArray = Annotated[
    NDArray[Shape[HeightWidth3Layers], float64],
    BeforeValidator(coerce_to_float64_array),
]  # type: ignore
UnitVector3DArray = Annotated[
    NDArray[Shape[UnitVector], float64],
    BeforeValidator(coerce_to_float64_array),
]  # type: ignore

__all__ = [
    "ScanMap2DArray",
    "ScanTensor3DArray",
    "ScanVectorField2DArray",
    "UnitVector3DArray",
]
