from typing import Annotated, Literal

from numpy import float64, uint8
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict


class FrozenBaseModel(BaseModel):
    """Base class for frozen Pydantic models."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


# Type aliases for array dimensions with shape constraints
FloatArray2D = Annotated[
    NDArray[Literal["*, *"], float64], "2D float array for depth data (surfaces)"
]
FloatArray1D = Annotated[
    NDArray[Literal["*"], float64], "1D float array for depth data (profiles)"
]
ImageArray2D = Annotated[NDArray[Literal["*, *"], uint8], "2D grayscale image"]
ImageArray3D = Annotated[NDArray[Literal["*, *, 3"], uint8], "3D RGB image (H×W×3)"]
