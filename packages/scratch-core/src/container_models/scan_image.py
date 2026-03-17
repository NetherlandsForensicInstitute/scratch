from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import Field
from scipy.constants import micro

from parsers import save_x3p, parse_to_x3p
from parsers.loaders import _load_surface
from .base import ConfigBaseModel, BinaryMask, FloatArray1D, DepthData


class ScanImage(ConfigBaseModel):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    data: DepthData
    scale_x: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    scale_y: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    meta_data: dict = Field(default_factory=dict)

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self.data.shape[0]

    @cached_property
    def valid_mask(self) -> BinaryMask:
        """Mask of the valid pixels in the data."""
        valid_mask = ~np.isnan(self.data)
        valid_mask.setflags(write=False)
        return valid_mask

    @cached_property
    def valid_data(self) -> FloatArray1D:
        """Valid pixels in the data."""
        valid_data = self.data[self.valid_mask]
        valid_data.setflags(write=False)
        return valid_data

    @property
    def center_meters(self) -> tuple[float, float]:
        """Get the image center in meters."""
        # TODO: Can we remove this?
        return self.width / 2 * self.scale_x, self.height / 2 * self.scale_y

    @classmethod
    def from_file(cls, scan_file: Path) -> Self:
        """
        Load a scan image from a file. Parsed values will be converted to meters (m).
        :param scan_file: The path to the file containing the scanned image data.
        :returns: An instance of `ScanImage`.
        """
        surface = _load_surface(scan_file)
        data = np.asarray(surface.data, dtype=np.float64) * micro
        step_x = surface.step_x * micro
        step_y = surface.step_y * micro

        return cls(
            data=data,
            scale_x=step_x,
            scale_y=step_y,
            meta_data=surface.metadata,
        )

    def export_to_x3p(self, output_path: Path) -> Path:
        """
        Convert a scan image to X3P format and save it to the specified path.

        :param parsed_scan: The scan image data to convert to X3P format.
        :param output_path: The file path where the X3P file will be saved.
        :return: The path to the saved X3P file.
        """
        return save_x3p(parse_to_x3p(self), output_path=output_path)
