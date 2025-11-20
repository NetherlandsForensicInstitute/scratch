import numpy as np
from numpydantic import NDArray, Shape, DType
from pydantic import BaseModel, Field


HeightWidth = NDArray[Shape["H W"], DType[np.floating]]
HeightWidthN = NDArray[Shape["H W N"], DType[np.floating]]
NormalVector = NDArray[Shape["N3"], DType[np.floating]]


class DepthMap(BaseModel):
    """A 2D depth map (height × width)."""

    data: NDArray[Shape[HeightWidth], DType[np.floating]] = Field(
        ..., description="2D depth map array of shape (H, W)"
    )


class SurfaceNormals(BaseModel):
    """Per-pixel unit surface normal components."""

    nx: NDArray[Shape[HeightWidth], DType[np.floating]] = Field(
        ..., description="Normal x-component"
    )

    ny: NDArray[Shape[HeightWidth], DType[np.floating]] = Field(
        ..., description="Normal y-component"
    )

    nz: NDArray[Shape[HeightWidth], DType[np.floating]] = Field(
        ..., description="Normal z-component"
    )

    def stack(self) -> NDArray:
        """Return a stacked (H × W × 3) normal map."""
        return np.stack([self.nx, self.ny, self.nz], axis=-1)


class SurfaceIntensity(BaseModel):
    """2D surface intensity map normalized to [0–255]."""

    intensity: NDArray[Shape[HeightWidth], DType[np.floating]] = Field(
        ..., description="2D intensity array (H, W)"
    )


class LightVector(BaseModel):
    """A normalized 3-element vector (x, y, z)."""

    vec: NDArray[Shape[NormalVector], DType[np.floating]] = Field(
        ..., description="Length-3 normalized direction vector"
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }


class LightingStack(BaseModel):
    """Stack of lighting intensity maps from multiple light sources."""

    intensity_stack: NDArray[Shape[HeightWidthN], DType[np.floating]] = Field(
        ..., description="Lighting stack with shape (H, W, N)"
    )

    @property
    def combined(self) -> NDArray:
        """Combine stacked lights → (H × W)."""
        return np.nansum(self.intensity_stack, axis=-1)
