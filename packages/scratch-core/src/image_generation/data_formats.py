import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from image_generation.translations import (
    normalize_intensity_map,
    apply_multiple_lights,
    compute_surface_normals,
)
from utils.array_definitions import (
    IMAGE_2D_ARRAY,
    IMAGE_3D_ARRAY,
    IMAGE_3_LAYER_STACK_ARRAY,
    NORMAL_VECTOR,
)


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )


class LightAngle(BaseModel):
    azimuth: float = Field(..., description="Azimuth angle in degrees.")
    elevation: float = Field(..., description="Elevation angle in degrees.")

    @property
    def vector(self):
        azr = np.deg2rad(self.azimuth)
        elr = np.deg2rad(self.elevation)
        v = np.array(
            [-np.cos(azr) * np.cos(elr), np.sin(azr) * np.cos(elr), np.sin(elr)]
        )
        return v


class Image2DArray(BaseModelConfig):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    data: IMAGE_2D_ARRAY = Field(..., description="2D depth map array of shape (H, W)")

    def compute_normals(
        self, x_dimension: float, y_dimension: float
    ) -> "SurfaceNormals":
        return SurfaceNormals(
            data=compute_surface_normals(self.data, x_dimension, y_dimension)
        )

    def normalize(self, max_val: float = 255, scale_min: float = 25) -> "Image2DArray":
        return Image2DArray(
            data=normalize_intensity_map(
                self.data, max_val=max_val, scale_min=scale_min
            )
        )


class Image3DArray(BaseModelConfig):
    """
    A 3D stack of 2D images.

    Used for: multiple lighting angles, multichannel data.
    Shape: (height, width, n_layers)
    """

    data: IMAGE_3D_ARRAY = Field(..., description="(H, W, N)")

    @property
    def combined(self) -> Image2DArray:
        """Combine stacked lights → (H × W)."""
        return Image2DArray(data=np.nansum(self.data, axis=2))


class SurfaceNormals(BaseModelConfig):
    """Normal vector at each pixel: shape (H, W, 3)."""

    data: IMAGE_3_LAYER_STACK_ARRAY

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

    def apply_lights(
        self,
        light_angles: tuple[NORMAL_VECTOR, ...],
        observer: NORMAL_VECTOR = LightAngle(azimuth=0, elevation=90).vector,
    ) -> "Image3DArray":
        return Image3DArray(
            data=apply_multiple_lights(self.data, light_angles, observer)
        )
