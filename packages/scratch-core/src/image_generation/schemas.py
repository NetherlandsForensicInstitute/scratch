import numpy as np
from pydantic import BaseModel, Field


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
