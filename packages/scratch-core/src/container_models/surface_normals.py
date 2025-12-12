from typing import Self
from pydantic import model_validator
from .base import ScanVectorField2DArray, ConfigBaseModel


class SurfaceNormals(ConfigBaseModel):
    """
    Normal vectors per pixel in a 3-layer field.

    Represents a surface-normal map with components (nx, ny, nz) stored in the
    last dimension. Shape: (height, width, 3).
    """

    x_normal_vector: ScanVectorField2DArray
    y_normal_vector: ScanVectorField2DArray
    z_normal_vector: ScanVectorField2DArray

    @model_validator(mode="after")
    def validate_same_shape(self) -> Self:
        """Validate that all normal vector components have the same shape."""
        x_shape = self.x_normal_vector.shape
        y_shape = self.y_normal_vector.shape
        z_shape = self.z_normal_vector.shape

        if not (x_shape == y_shape == z_shape):
            raise ValueError(
                f"All normal vector components must have the same shape. "
                f"Got x: {x_shape}, y: {y_shape}, z: {z_shape}"
            )

        return self
