from numpydantic import NDArray, Shape
from typing import Annotated

HeightWidth = "*, *"
HeightWidthNLayers = f"{HeightWidth}, *"
HeightWidth3Layers = f"{HeightWidth}, 3"
NormalVector = "3"

IMAGE_2D_ARRAY = Annotated[NDArray[Shape[HeightWidth], float], ...]  # type: ignore
IMAGE_3D_ARRAY = Annotated[NDArray[Shape[HeightWidthNLayers], float], ...]  # type: ignore
IMAGE_3_LAYER_STACK_ARRAY = Annotated[NDArray[Shape[HeightWidth3Layers], float], ...]  # type: ignore
NORMAL_VECTOR = Annotated[NDArray[Shape[NormalVector], float], ...]  # type: ignore
