from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
from PIL.Image import Image, fromarray
from pydantic import Field
from scipy.constants import micro

from computations.normalization import _grayscale_to_rgba, _normalize_2d_array
from parsers import convert_to_x3p, save_x3p
from parsers.loaders import _load_surface

from .base import (
    BinaryMask,
    ConfigBaseModel,
    DepthData,
    FloatArray1D,
)
from .models import NormalizationBounds


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

    def _to_pil_image(self, normalization_bounds: NormalizationBounds) -> Image:
        """Get a rgba image from the scan data."""
        return fromarray(
            _grayscale_to_rgba(
                scan_data=_normalize_2d_array(
                    self.data, normalization_bounds=normalization_bounds
                )
            )
        )

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

    def save_as_x3p(self, output_path: Path) -> None:
        """
        Convert a scan image to X3P format and save it to the specified path.

        :param parsed_scan: The scan image data to convert to X3P format.
        :param output_path: The file path where the X3P file will be saved.
        """
        save_x3p(convert_to_x3p(self), output_path=output_path)

    def save_as_image(
        self, output_path: Path, normalization_bounds: NormalizationBounds
    ):
        """
        Convert ScanImage data to an Image and save it to the given output_path.

        :param output_path: the given path to save the scan data.
        :param normalization_bounds: the scaling needed for normalizing the ScanImage.
        :return: the output path to where the image is saved.
        """
        self._to_pil_image(normalization_bounds=normalization_bounds).save(output_path)
