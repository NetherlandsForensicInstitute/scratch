import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field, model_validator, ConfigDict

HeightWidth = "*, *"
HeightWidthN = "*, *, *"
NormalVector = "3"


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )


class DepthMap(BaseModelConfig):
    """A 2D depth map (height × width)."""

    data: NDArray[Shape[HeightWidth], float] = Field(
        ..., description="2D depth map array of shape (H, W)"
    )
    xdim: float = Field(
        ..., description="X dimension, the space between pixels in meters."
    )
    ydim: float = Field(
        ..., description="Y dimension, the space between pixels in meters."
    )


class SurfaceNormals(BaseModelConfig):
    """Per-pixel unit surface normal components."""

    nx: NDArray[Shape[HeightWidth], float] = Field(
        ..., description="Normal x-component"
    )

    ny: NDArray[Shape[HeightWidth], float] = Field(
        ..., description="Normal y-component"
    )

    nz: NDArray[Shape[HeightWidth], float] = Field(
        ..., description="Normal z-component"
    )

    @model_validator(mode="after")
    def validate_shapes(self):
        """Ensure nx, ny, nz all share the same H × W dimensions."""
        if not (self.nx.shape == self.ny.shape == self.nz.shape):
            raise ValueError(
                f"SurfaceNormals must have matching shapes, "
                f"got nx={self.nx.shape}, ny={self.ny.shape}, nz={self.nz.shape}"
            )
        return self

    def stack(self) -> NDArray:
        """Return a stacked (H × W × 3) normal map."""
        return np.stack([self.nx, self.ny, self.nz], axis=-1)


class SurfaceIntensity(BaseModelConfig):
    """2D surface intensity map normalized to [0–255]."""

    intensity: NDArray[Shape[HeightWidth], float] = Field(
        ..., description="2D intensity array (H, W)"
    )


class LightVector(BaseModelConfig):
    """A normalized 3-element vector (x, y, z)."""

    vec: NDArray[Shape[NormalVector], float] = Field(
        ..., description="Length-3 normalized direction vector"
    )


class LightingStack(BaseModelConfig):
    """Stack of lighting intensity maps from multiple light sources."""

    intensity_stack: NDArray[Shape[HeightWidthN], float] = Field(
        ..., description="Lighting stack with shape (H, W, N)"
    )

    @property
    def combined(self) -> NDArray:
        """Combine stacked lights → (H × W)."""
        return np.nansum(self.intensity_stack, axis=2)
