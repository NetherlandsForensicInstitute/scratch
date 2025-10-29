from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

Array2D = np.ndarray[tuple[int, int], np.float32 | np.float64]


class FrozenBaseModel(BaseModel):
    """Base class for frozen Pydantic models."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class ParsedImage(FrozenBaseModel):
    """Class for storing parsed scan data."""

    data: Array2D
    scale_x: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    scale_y: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    path: Path | None = None
    metadata: dict | None = None

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width_um(self) -> float:
        return self.scale_x * self.width

    @property
    def height_um(self) -> float:
        return self.scale_y * self.height
