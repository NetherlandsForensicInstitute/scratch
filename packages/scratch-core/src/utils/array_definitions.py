from numpydantic import NDArray, Shape
from numpy import float64
from typing import Annotated

HeightWidth = "*, *"
HeightWidthNLayers = f"{HeightWidth}, *"
HeightWidth3Layers = f"{HeightWidth}, 3"
NormalVector = "3"

ScanMap2D = Annotated[NDArray[Shape[HeightWidth], float64], ...]  # type: ignore
ScanTensor3D = Annotated[NDArray[Shape[HeightWidthNLayers], float64], ...]  # type: ignore
ScanVectorField2D = Annotated[NDArray[Shape[HeightWidth3Layers], float64], ...]  # type: ignore
Vector3D = Annotated[NDArray[Shape[NormalVector], float64], ...]  # type: ignore
