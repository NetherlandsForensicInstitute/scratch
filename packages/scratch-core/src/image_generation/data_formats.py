import numpy as np
from pydantic import BaseModel, Field, ConfigDict, RootModel

from image_generation.translations import (
    normalize_2d_array,
    apply_multiple_lights,
    compute_surface_normals,
)
from utils.array_definitions import (
    ScanMap2DArray,
    ScanTensor3DArray,
    ScanVectorField2DArray,
    Vector3DArray,
)


class LightSource(BaseModel):
    """
    Representation of a light source using an angular direction (azimuth and elevation)
    together with a derived 3D unit direction vector.

    Angle conventions:
    - Azimuth: horizontal angle measured in the x–y plane.
      0° corresponds to the –x direction, and the angle increases counter-clockwise
      toward the +y direction.

    - Elevation: vertical angle measured relative to the x–y plane.
      0° is horizontal, +90° points straight upward (+z), and –90° points straight
      downward (–z).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
    azimuth: float = Field(
        ...,
        description="Horizontal direction angle in degrees. "
        "Measured in the x–y plane from the –x axis, increasing counter-clockwise.",
        examples=[90, 45, 180],
        ge=0,
        le=360,
    )
    elevation: float = Field(
        ...,
        description="Vertical angle in degrees measured from the x–y plane. "
        "0° is horizontal, +90° is upward (+z), –90° is downward (–z).",
        examples=[90, 45, 180],
        ge=-90,
        le=90,
    )

    @property
    def unit_vector(self) -> Vector3DArray:
        """
        Returns the unit direction vector [x, y, z] corresponding to the azimuth and
        elevation angles. The conversion follows a spherical-coordinate convention:
        azimuth defines the horizontal direction, and elevation defines the vertical
        tilt relative to the x–y plane.
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


class ScanMap2D(RootModel[ScanMap2DArray]):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    def compute_normals(
        self, x_dimension: float, y_dimension: float
    ) -> "SurfaceNormals":
        """
        Compute per-pixel surface normals from a 2D depth map.

        :param x_dimension: Represents the distance between 2 pixels in meters in x direction.
        :param y_dimension: Represents the distance between 2 pixels in meters in x direction.

        :returns: Normal vectors per pixel in a 3-layer field. Layers are [x,y,z]
        """
        return SurfaceNormals(
            compute_surface_normals(self.root, x_dimension, y_dimension)
        )

    def normalize(self, scale_max: float = 255, scale_min: float = 25) -> "ScanMap2D":
        """
        Normalize a 2D intensity map to a specified output range.

        :param scale_max: Maximum output intensity value. Default is ``255``.
        :param scale_min: Minimum output intensity value. Default is ``25``.

        :returns: Normalized 2D intensity map with values in ``[scale_min, max_val]``.
        """
        return ScanMap2D(
            normalize_2d_array(self.root, scale_max=scale_max, scale_min=scale_min)
        )


class ScanTensor3D(RootModel[ScanTensor3DArray]):
    """
    A 3D stack of 2D scan maps.

    Typically used for multi-illumination data or multichannel measurements.
    Shape: (height, width, n_layers)
    """

    @property
    def combined(self) -> "ScanMap2D":
        """Combine stacked lights → (Height × Width)."""
        return ScanMap2D(np.nansum(self.root, axis=2))


class SurfaceNormals(RootModel[ScanVectorField2DArray]):
    """Normal vectors per pixel in a 3-layer field.

    Represents a surface-normal map with components (nx, ny, nz) stored in the
    last dimension. Shape: (height, width, 3)."""

    def apply_lights(
        self,
        light_vectors: tuple[Vector3DArray, ...],
        observer: Vector3DArray = LightSource(azimuth=0, elevation=90).unit_vector,
    ) -> "ScanTensor3D":
        """
        Apply one or more light vectors to the surface-normal field.

        Computes intensity values for each light direction (and an optional observer
        direction) and returns a stacked result as an Image3DArray.

        :param light_vectors: LightSource objects defining azimuth and elevation as a unit vector.
        :param observer: LightSource object defining azimuth and elevation as a unit vector as the observer.
            Defaults to azimuth=0, elevation=90

        :returns: Normalized 2D intensity map with shape (Height, Width), suitable for
        """
        return ScanTensor3D(
            apply_multiple_lights(
                surface_normals=self.root,
                light_vectors=light_vectors,
                observer_vector=observer,
            )
        )
