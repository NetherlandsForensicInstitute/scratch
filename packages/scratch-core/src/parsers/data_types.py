from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from PIL import Image
import numpy as np
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC
from .patches.al3d import read_al3d

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


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
            raise ValueError(f"Invalid file extension: {ext}")
    return path


def validate_array_shape(array: NDArray) -> NDArray:
    """Test whether the passed array has a valid shape."""
    if len(array.shape) != 2:
        raise ValueError(f"Invalid array shape: {array.shape}")
    return array


class ParsedImage(FrozenBaseModel):
    """
    Class for storing parsed scan data.

    :param data: A numpy array containing the parsed 2D image data.
    :param scale_x: The pixel size in the X-direction in micrometers (um).
    :param scale_y: The pixel size in the Y-direction in micrometers (um).
    :param path_to_original_image: The filepath to the original image.
    :param meta_data: (Optional) A dictionary containing the metadata.
    """

    data: Annotated[NDArray, AfterValidator(validate_array_shape)]
    scale_x: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    scale_y: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    path_to_original_image: Path
    meta_data: dict | None = None

    @classmethod
    def from_file(cls, path: Path) -> "ParsedImage":
        extension = path.suffix.lower()[1:]
        if extension in ScanFileFormats:
            surface = Surface.load(path)
            return ParsedImage(
                data=np.asarray(surface.data, dtype=np.float64),
                scale_x=surface.step_x,
                scale_y=surface.step_y,
                meta_data=surface.metadata,
                path_to_original_image=path,
            )
        elif extension in ImageFileFormats:
            return ParsedImage(
                data=np.asarray(
                    Image.open(path).convert("L"),
                    dtype=np.float64,
                ),
                path_to_original_image=path,
            )
        else:
            raise ValueError(f"Invalid file extension: {extension}")

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
