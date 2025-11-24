import numpy as np
from pydantic import BaseModel, Field, ConfigDict, RootModel

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


class LightAngle(BaseModel):
    """Representation of a light angle and the thereby proprty a vector of [x,y,z]"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
    azimuth: float = Field(..., description="Azimuth angle in degrees.")
    elevation: float = Field(..., description="Elevation angle in degrees.")

    @property
    def vector(self) -> NORMAL_VECTOR:
        azimuth = np.deg2rad(self.azimuth)
        elevation = np.deg2rad(self.elevation)
        return np.array(
            [
                -np.cos(azimuth) * np.cos(elevation),
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
            ]
        )


class Image2DArray(RootModel[IMAGE_2D_ARRAY]):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    def compute_normals(self, x_dimension: float, y_dimension: float):
        return SurfaceNormals(
            compute_surface_normals(self.root, x_dimension, y_dimension)
        )

    def normalize(self, max_val: float = 255, scale_min: float = 25):
        return Image2DArray(
            normalize_intensity_map(self.root, max_val=max_val, scale_min=scale_min)
        )


class Image3DArray(RootModel[IMAGE_3D_ARRAY]):
    """
    A 3D stack of 2D images.

    Used for: multiple lighting angles, multichannel data.
    Shape: (height, width, n_layers)
    """

    @property
    def combined(self) -> Image2DArray:
        """Combine stacked lights → (Height × Width)."""
        return Image2DArray(np.nansum(self.root, axis=2))


class SurfaceNormals(RootModel[IMAGE_3_LAYER_STACK_ARRAY]):
    """Normal vector at each pixel: shape (Height, Width, 3-layers(x,y,z))."""

    @property
    def nx(self) -> IMAGE_2D_ARRAY:
        return self.root[..., 0]

    @property
    def ny(self) -> IMAGE_2D_ARRAY:
        return self.root[..., 1]

    @property
    def nz(self) -> IMAGE_2D_ARRAY:
        return self.root[..., 2]

    @classmethod
    def from_components(
        cls,
        nx: IMAGE_2D_ARRAY,
        ny: IMAGE_2D_ARRAY,
        nz: IMAGE_2D_ARRAY,
    ) -> "SurfaceNormals":
        """Construct from separate component arrays."""
        return cls(np.stack([nx, ny, nz], axis=-1))

    def apply_lights(
        self,
        light_vectors: tuple[NORMAL_VECTOR, ...],
        observer: NORMAL_VECTOR = LightAngle(azimuth=0, elevation=90).vector,
    ) -> "Image3DArray":
        return Image3DArray(
            apply_multiple_lights(
                surface_normals=self.root,
                light_vectors=light_vectors,
                observer_vector=observer,
            )
        )
