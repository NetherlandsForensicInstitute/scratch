import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field, ConfigDict

HeightWidth = "*, *"
HeightWidthNLayers = f"{HeightWidth}, *"
HeightWidth3Layers = f"{HeightWidth}, 3"
NormalVector = "3"

IMAGE_2D_ARRAY = NDArray[Shape[HeightWidth], float]
IMAGE_3D_ARRAY = NDArray[Shape[HeightWidthNLayers], float]
IMAGE_3_STACK_ARRAY = NDArray[Shape[HeightWidth3Layers], float]
NORMAL_VECTOR = NDArray[Shape[NormalVector], float]


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )


class Image2DArray(BaseModelConfig):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    data: IMAGE_2D_ARRAY = Field(..., description="2D depth map array of shape (H, W)")


class Image3DArray(BaseModelConfig):
    """
    A 3D stack of 2D images.

    Used for: multiple lighting angles, multi-channel data.
    Shape: (height, width, n_layers)
    """

    data: IMAGE_3D_ARRAY = Field(..., description="(H, W, N)")


class SurfaceNormals(BaseModelConfig):
    """Normal vector at each pixel: shape (H, W, 3)."""

    data: IMAGE_3_STACK_ARRAY

    @property
    def nx(self) -> IMAGE_2D_ARRAY:
        return self.data[..., 0]

    @property
    def ny(self) -> IMAGE_2D_ARRAY:
        return self.data[..., 1]

    @property
    def nz(self) -> IMAGE_2D_ARRAY:
        return self.data[..., 2]

    @classmethod
    def from_components(
        cls,
        nx: IMAGE_2D_ARRAY,
        ny: IMAGE_2D_ARRAY,
        nz: IMAGE_2D_ARRAY,
    ) -> "SurfaceNormals":
        """Construct from separate component arrays."""
        return cls(data=np.stack([nx, ny, nz], axis=-1))


class LightVector(BaseModelConfig):
    """A normalized 3-element vector (x, y, z)."""

    vec: NORMAL_VECTOR = Field(..., description="Length-3 normalized direction vector")


class LightingStack(Image3DArray):
    """Stack of lighting intensity maps from multiple light sources."""

    @property
    def combined(self) -> Image2DArray:
        """Combine stacked lights → (H × W)."""
        return Image2DArray(data=np.nansum(self.data, axis=2))
