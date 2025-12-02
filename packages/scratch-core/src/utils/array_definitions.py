from collections.abc import Sequence
from functools import partial
from typing import Annotated, Any, TypeAlias
from numpy import bool_, float64, uint8, array
from numpy.typing import DTypeLike, NDArray
from pydantic import BeforeValidator, PlainSerializer


def _serialize_ndarray(array_: NDArray[Any]) -> list:
    """Serialize numpy array to a Python list for JSON serialization."""
    return array_.tolist()


def _coerce_to_array(
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


ScanMapRGBA: TypeAlias = Annotated[
    NDArray[uint8],
    BeforeValidator(partial(_coerce_to_array, uint8)),
    PlainSerializer(_serialize_ndarray),
]

ScanTensor3DArray = ScanMap2DArray = ScanVectorField2DArray = UnitVector3DArray = (
    Annotated[
        NDArray[float64],
        BeforeValidator(partial(_coerce_to_array, float64)),
        PlainSerializer(_serialize_ndarray),
    ]
)
MaskArray = Annotated[
    NDArray[bool_],
    BeforeValidator(partial(_coerce_to_array, bool_)),
    PlainSerializer(_serialize_ndarray),
]
