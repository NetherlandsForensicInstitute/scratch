from pydantic import BaseModel, Field, model_validator


class IntensityScaling(BaseModel):
    """Representing the upper and lower scaling for the pixel intensity in Normalizations"""

    scale_max: int = Field(..., ge=0, le=255)
    scale_min: int = Field(..., ge=0)

    @model_validator(mode="after")
    def check_scale_order(self):
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        return self
