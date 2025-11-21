from numpydantic import NDArray, Shape

HeightWidth = "*, *"
HeightWidthNLayers = f"{HeightWidth}, *"
HeightWidth3Layers = f"{HeightWidth}, 3"
NormalVector = "3"

IMAGE_2D_ARRAY = NDArray[Shape[HeightWidth], float]
IMAGE_3D_ARRAY = NDArray[Shape[HeightWidthNLayers], float]
IMAGE_3_LAYER_STACK_ARRAY = NDArray[Shape[HeightWidth3Layers], float]
NORMAL_VECTOR = NDArray[Shape[NormalVector], float]
