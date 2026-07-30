from pydantic import BaseModel, Field, model_validator

from container_models.base import FloatArray2D


class NormalizationBounds(BaseModel):
    """Pixel intensity bounds within the [0, 255] grayscale range."""

    low: int = Field(..., ge=0, le=255)
    high: int = Field(..., ge=0, le=255)

    @model_validator(mode="after")
    def check_scale_order(self):
        if self.high <= self.low:
            raise ValueError("scale_max must be greater than scale_min")
        return self


class LevelingResult(BaseModel, arbitrary_types_allowed=True):
    """
    Result of a leveling operation.

    :param leveled_map: 2D array with the leveled height data
    :param fitted_surface: 2D array of the fitted surface (same shape as `leveled_map`)
    """

    leveled_map: FloatArray2D
    fitted_surface: FloatArray2D
