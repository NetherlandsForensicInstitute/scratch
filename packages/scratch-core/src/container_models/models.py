from pydantic import BaseModel, Field, model_validator


class NormalizationBounds(BaseModel):
    """Pixel intensity bounds within the [0, 255] grayscale range."""

    low: int = Field(..., ge=0, le=255)
    high: int = Field(..., ge=0, le=255)

    @model_validator(mode="after")
    def check_scale_order(self):
        if self.high <= self.low:
            raise ValueError("scale_max must be greater than scale_min")
        return self
