from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
import numpy as np
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC
from .patches.al3d import read_al3d

UNIT_CONVERSION_FACTOR = 1e-6  # conversion factor from micrometers (um) to meters (m)

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)

Array2D = NDArray[tuple[int, int]]


class ScanFileFormats(StrEnum):
    AL3D = auto()
    SUR = auto()
    X3P = auto()
    PLU = auto()


class FrozenBaseModel(BaseModel):
    """Base class for frozen Pydantic models."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


def validate_parsed_image_shape(image_data: NDArray) -> NDArray:
    """Test whether the parsed image data has a valid shape."""
    if len(image_data.shape) != 2:
        raise ValueError(f"Invalid array shape: {image_data.shape}")
    return image_data


class ScanImage(FrozenBaseModel):
    """
    Class for storing parsed scan data.

    The image data is stored as a 2D floating point tensor with shape `[height, width]`.

    :param data: A numpy array containing the parsed 2D image data.
    :param scale_x: The pixel size in the X-direction in meters (m).
    :param scale_y: The pixel size in the Y-direction in meters (m).
    :param path_to_original_image: The filepath to the original image.
    :param is_subsampled: Whether the image data is subsampled from the original image.
    :param meta_data: (Optional) A dictionary containing the metadata.
    """

    data: Annotated[Array2D, AfterValidator(validate_parsed_image_shape)]
    scale_x: float = Field(default=1.0, gt=0.0, description="pixel size in meters (m)")
    scale_y: float = Field(default=1.0, gt=0.0, description="pixel size in meters (m)")
    path_to_original_image: Path
    is_subsampled: bool = False
    meta_data: dict | None = None

    @classmethod
    def from_file(cls, scan_file: Path) -> "ScanImage":
        """
        Load a scan image from a file. Parsed values will be converted to meters (m).

        :param scan_file: The path to the file containing the scanned image data.
        :returns: An instance of `ScanImage` containing the parsed data.
        """
        if extension := scan_file.suffix.lower()[1:] not in ScanFileFormats:
            raise ValueError(f"Invalid file extension: {extension}")
        surface = Surface.load(scan_file)
        return ScanImage(
            data=np.asarray(surface.data, dtype=np.float64) * UNIT_CONVERSION_FACTOR,
            scale_x=surface.step_x * UNIT_CONVERSION_FACTOR,
            scale_y=surface.step_y * UNIT_CONVERSION_FACTOR,
            meta_data=surface.metadata,
            path_to_original_image=scan_file,
        )

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self.data.shape[0]

    @property
    def width_meters(self) -> float:
        """The image width in meters."""
        return self.scale_x * self.width

    @property
    def height_meters(self) -> float:
        """The image height in meters."""
        return self.scale_y * self.height
