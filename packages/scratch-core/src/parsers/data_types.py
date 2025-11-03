from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

Array2D = np.ndarray[tuple[int, int], np.float64]


class FileFormats(StrEnum):
    PNG = auto()
    AL3D = auto()
    X3P = auto()


class FrozenBaseModel(BaseModel):
    """Base class for frozen Pydantic models."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


def validate_file_extension(path: Path | None = None) -> Path | None:
    """Test whether the file extension is valid."""
    if path is not None:
        _ = FileFormats(path.suffix[1:])
    return path


class ParsedImage(FrozenBaseModel):
    """
    Class for storing parsed scan data.

    :param data: A numpy array containing the parsed 2D image data.
    :param scale_x: The pixel size in the X-direction in micrometers (um).
    :param scale_y: The pixel size in the Y-direction in micrometers (um).
    :path path_to_original_image: (Optional) The filepath to the original image.
    :param meta_data: (Optional) A dictionary containing the metadata.
    """

    data: Array2D
    scale_x: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    scale_y: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    path_to_original_image: Annotated[
        Path | None, AfterValidator(validate_file_extension)
    ] = None
    meta_data: dict | None = None

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self.data.shape[0]

    @property
    def width_um(self) -> float:
        """The image width in micrometers."""
        return self.scale_x * self.width

    @property
    def height_um(self) -> float:
        """The image height in micrometers."""
        return self.scale_y * self.height
