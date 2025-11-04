from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class ImageFileFormats(StrEnum):
    PNG = auto()
    BMP = auto()
    JPG = auto()
    JPEG = auto()
    TIF = auto()
    TIFF = auto()


class ScanFileFormats(StrEnum):
    AL3D = auto()
    SUR = auto()
    X3P = auto()
    PLU = auto()


class FrozenBaseModel(BaseModel):
    """Base class for frozen Pydantic models."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


def validate_file_extension(path: Path) -> Path:
    """Test whether the file extension is valid."""
    if path is not None:
        ext = path.suffix[1:]
        if not (ext in ImageFileFormats or ext in ScanFileFormats):
            raise ValueError("Invalid file extension")
    return path


class ParsedImage(FrozenBaseModel):
    """
    Class for storing parsed scan data.

    :param data: A numpy array containing the parsed 2D image data.
    :param scale_x: The pixel size in the X-direction in micrometers (um).
    :param scale_y: The pixel size in the Y-direction in micrometers (um).
    :param path_to_original_image: (Optional) The filepath to the original image.
    :param meta_data: (Optional) A dictionary containing the metadata.
    """

    data: NDArray
    scale_x: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    scale_y: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    path_to_original_image: Annotated[Path, AfterValidator(validate_file_extension)]
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
