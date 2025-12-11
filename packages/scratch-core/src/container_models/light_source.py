from functools import cached_property
import numpy as np
from pydantic import Field
from .base import UnitVector3DArray, ConfigBaseModel


class LightSource(ConfigBaseModel):
    """
    Representation of a light source using an angular direction (azimuth and elevation)
    together with a derived 3D unit direction vector.
    """

    azimuth: float = Field(
        ...,
        description="Horizontal angle in degrees measured from the –x axis in the x–y plane. "
        "0° is –x direction, 90° is +y direction, 180° is +x direction.",
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

    @cached_property
    def unit_vector(self) -> UnitVector3DArray:
        """
        Returns the unit direction vector [x, y, z] corresponding to the azimuth and
        elevation angles. The conversion follows a spherical-coordinate convention:
        azimuth defines the horizontal direction, and elevation defines the vertical
        tilt relative to the x–y plane.
        """
        azimuth = np.deg2rad(self.azimuth)
        elevation = np.deg2rad(self.elevation)
        vec = np.array(
            [
                -np.cos(azimuth) * np.cos(elevation),
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
            ]
        )
        vec.setflags(write=False)
        return vec
