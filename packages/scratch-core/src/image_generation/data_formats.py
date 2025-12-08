import numpy as np
from numpydantic.ndarray import NDArray
from PIL.Image import Image, fromarray
from pydantic import BaseModel, ConfigDict, Field
from loguru import logger
from .exceptions import ImageGenerationError
from conversion.exceptions import ConversionError
from conversion.subsample import subsample_array
from image_generation.translations import (
    apply_multiple_lights,
    compute_surface_normals,
    grayscale_to_rgba,
    normalize_2d_array,
)
from utils.array_definitions import (
    ScanMap2DArray,
    ScanTensor3DArray,
    ScanVectorField2DArray,
    UnitVector3DArray,
)


class ImageContainer(BaseModel):
    data: NDArray
    scale_x: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    scale_y: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    meta_data: dict | None = None


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
    def unit_vector(self) -> UnitVector3DArray:
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


class ScanImage(ImageContainer, arbitrary_types_allowed=True):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    data: ScanMap2DArray

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self.data.shape[0]

    def subsample(self, step_x: int, step_y: int) -> "ScanImage":
        """Subsample the data in a `ScanMap2D` instance by skipping `step_size` steps."""
        logger.debug(f"Subsampling data with step size ({step_x}, {step_y})")
        try:
            array = subsample_array(scan_data=self.data, step_size=(step_x, step_y))
        except ValueError as e:
            logger.error(f"Error subsampling data: {e}")
            raise ImageGenerationError(f"Error subsampling data: {e}") from e
        return ScanImage(
            data=array,
            scale_x=self.scale_x * step_x,
            scale_y=self.scale_y * step_y,
        )

    def compute_normals(
        self, x_dimension: float, y_dimension: float
    ) -> "SurfaceNormals":
        """
        Compute per-pixel surface normals from a 2D depth map.

        :param x_dimension: Represents the distance between 2 pixels in meters in x direction.
        :param y_dimension: Represents the distance between 2 pixels in meters in x direction.

        :returns: Normal vectors per pixel in a 3-layer field. Layers are [x,y,z]
        """
        try:
            return SurfaceNormals(
                data=compute_surface_normals(self.data, x_dimension, y_dimension),
                scale_x=self.scale_x,
                scale_y=self.scale_y,
            )
        except ValueError as e:
            logger.error(f"Error computing surface normals: {e}")
            raise ImageGenerationError(f"Error computing surface normals: {e}") from e

    def normalize(self, scale_max: float = 255, scale_min: float = 25) -> "ScanImage":
        """
        Normalize a 2D intensity map to a specified output range.

        :param scale_max: Maximum output intensity value. Default is ``255``.
        :param scale_min: Minimum output intensity value. Default is ``25``.

        :returns: Normalized 2D intensity map with values in ``[scale_min, max_val]``.
        """
        return ScanImage(
            data=normalize_2d_array(
                self.data, scale_max=scale_max, scale_min=scale_min
            ),
            scale_x=self.scale_x,
            scale_y=self.scale_y,
        )

    def image(self) -> Image:
        """
        Convert a 2D intensity map to an image.

        :returns: Image representation of the 2D intensity map.
        """
        try:
            return fromarray(grayscale_to_rgba(scan_data=self.data))
        except ValueError as err:
            raise ConversionError("Could not convert data to an RGBA image.") from err


class MultiIlluminationScan(ImageContainer, arbitrary_types_allowed=True):
    """
    Multiple 2D scans captured under different illumination conditions.

    Shape: (height, width, n_lights) where the last axis represents
    different lighting directions applied to the same surface.
    """

    data: ScanTensor3DArray

    def reduce_stack(self, merge_on_axis: int = 2) -> ScanImage:
        """Combine stacked 2d scan maps → (Height × Width)."""
        return ScanImage(
            data=np.nansum(self.data, axis=merge_on_axis),
            scale_x=self.scale_x,
            scale_y=self.scale_y,
        )


class SurfaceNormals(ImageContainer, arbitrary_types_allowed=True):
    """Normal vectors per pixel in a 3-layer field.

    Represents a surface-normal map with components (nx, ny, nz) stored in the
    last dimension. Shape: (height, width, 3)."""

    data: ScanVectorField2DArray

    def apply_lights(
        self,
        light_vectors: tuple[UnitVector3DArray, ...],
        observer: UnitVector3DArray = LightSource(azimuth=0, elevation=90).unit_vector,
    ) -> "MultiIlluminationScan":
        """
        Apply one or more light vectors to the surface-normal field.

        Computes intensity values for each light direction (and an optional observer
        direction) and returns a stacked result as an Image3DArray.

        :param light_vectors: LightSource objects defining azimuth and elevation as a unit vector.
        :param observer: LightSource object defining azimuth and elevation as a unit vector as the observer.
            Defaults to azimuth=0, elevation=90

        :returns: Normalized 2D intensity map with shape (Height, Width), suitable for
        """
        return MultiIlluminationScan(
            data=apply_multiple_lights(
                surface_normals=self.data,
                light_vectors=light_vectors,
                observer_vector=observer,
            ),
            scale_x=self.scale_x,
            scale_y=self.scale_y,
        )
