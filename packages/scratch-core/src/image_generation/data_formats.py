import numpy as np
from pydantic import BaseModel, Field, ConfigDict, RootModel

from image_generation.translations import (
    normalize_intensity_map,
    apply_multiple_lights,
    compute_surface_normals,
)
from utils.array_definitions import (
    ScanMap2D,
    ScanTensor3D,
    ScanVectorField2D,
    Vector3D,
)


class LightSource(BaseModel):
    """Representation of a light angle and the thereby proprty a vector of [x,y,z]"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
    azimuth: float = Field(
        ...,
        description="Azimuth angle in degrees.",
        examples=[90, 45, 180],
        ge=0,
        le=360,
    )
    elevation: float = Field(
        ...,
        description="Elevation angle in degrees.",
        examples=[90, 45, 180],
        ge=0,
        le=360,
    )

    @property
    def vector(self) -> Vector3D:
        """
        The `vector` property is a unit direction vector in 3D space as a NumPy array of the form `[x, y, z]`, derived from the
        angular representation.
        """
        azimuth = np.deg2rad(self.azimuth)
        elevation = np.deg2rad(self.elevation)
        return np.array(
            [
                -np.cos(azimuth) * np.cos(elevation),
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
            ]
        )


class Image2DArray(RootModel[ScanMap2D]):
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


class Image3DArray(RootModel[ScanTensor3D]):
    """
    A 3D stack of 2D scan maps.

    Typically used for multi-illumination data or multichannel measurements.
    Shape: (height, width, n_layers)
    """

    @property
    def combined(self) -> Image2DArray:
        """Combine stacked lights → (Height × Width)."""
        return Image2DArray(np.nansum(self.root, axis=2))


class SurfaceNormals(RootModel[ScanVectorField2D]):
    """Normal vectors per pixel in a 3-layer field.

    Represents a surface-normal map with components (nx, ny, nz) stored in the
    last dimension. Shape: (height, width, 3)."""

    @property
    def nx(self) -> ScanMap2D:
        """ "X-component of the surface normal as a 2D scan map."""
        return self.root[..., 0]

    @property
    def ny(self) -> ScanMap2D:
        """Y-component of the surface normal as a 2D scan map."""
        return self.root[..., 1]

    @property
    def nz(self) -> ScanMap2D:
        """Z-component of the surface normal as a 2D scan map."""
        return self.root[..., 2]

    @classmethod
    def from_components(
        cls,
        nx: ScanMap2D,
        ny: ScanMap2D,
        nz: ScanMap2D,
    ) -> "SurfaceNormals":
        """Create a SurfaceNormals object from separate nx, ny, and nz 2D component maps."""
        return cls(np.stack([nx, ny, nz], axis=-1))

    def apply_lights(
        self,
        light_vectors: tuple[Vector3D, ...],
        observer: Vector3D = LightSource(azimuth=0, elevation=90).vector,
    ) -> "Image3DArray":
        """
        Apply one or more light vectors to the surface-normal field.

        Computes intensity values for each light direction (and an optional observer
        direction) and returns a stacked result as an Image3DArray.
        """
        return Image3DArray(
            apply_multiple_lights(
                surface_normals=self.root,
                light_vectors=light_vectors,
                observer_vector=observer,
            )
        )
