from pydantic import BaseModel, Field, model_validator


class ImageScaling(BaseModel):
    scale_max: float = Field(..., le=255)
    scale_min: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_scale_order(self):
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        return self
