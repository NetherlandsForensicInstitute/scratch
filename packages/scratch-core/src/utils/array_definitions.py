from numpydantic import NDArray, Shape
from numpy import float64
from typing import Annotated

HeightWidthShape = "*, *"
HeightWidthNLayers = f"{HeightWidthShape}, *"
HeightWidth3Layers = f"{HeightWidthShape}, 3"
NormalVector = "3"

ScanMap2DArray = Annotated[NDArray[Shape[HeightWidthShape], float64], ...]  # type: ignore
ScanTensor3DArray = Annotated[NDArray[Shape[HeightWidthNLayers], float64], ...]  # type: ignore
ScanVectorField2DArray = Annotated[NDArray[Shape[HeightWidth3Layers], float64], ...]  # type: ignore
Vector3DArray = Annotated[NDArray[Shape[NormalVector], float64], ...]  # type: ignore
