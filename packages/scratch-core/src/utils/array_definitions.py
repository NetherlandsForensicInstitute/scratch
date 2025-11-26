from numpydantic import NDArray, Shape
from numpy import float64
from typing import Annotated

HeightWidthShape = "*, *"
HeightWidthNLayers = f"{HeightWidthShape}, *"
HeightWidth3Layers = f"{HeightWidthShape}, 3"
UnitVector = "3"

ScanMap2DArray = Annotated[NDArray[Shape[HeightWidthShape], float64], ...]  # type: ignore
ScanTensor3DArray = Annotated[NDArray[Shape[HeightWidthNLayers], float64], ...]  # type: ignore
ScanVectorField2DArray = Annotated[NDArray[Shape[HeightWidth3Layers], float64], ...]  # type: ignore
UnitVector3DArray = Annotated[NDArray[Shape[UnitVector], float64], ...]  # type: ignore

__all__ = [
    "ScanMap2DArray",
    "ScanTensor3DArray",
    "ScanVectorField2DArray",
    "UnitVector3DArray",
]
