from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
from PIL.Image import Image, fromarray
from pydantic import Field
from scipy.constants import micro

from parsers import convert_to_x3p, save_x3p
from parsers.loaders import _load_surface

from .base import (
    BinaryMask,
    ConfigBaseModel,
    DepthData,
    FloatArray1D,
    FloatArray2D,
    ImageRGBA,
)


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

    def _to_pil_image(self, scale_max: float, scale_min: float) -> Image:
        """Get a rgba image from the scan data."""
        return fromarray(
            grayscale_to_rgba(
                scan_data=normalize_2d_array(
                    self.data, scale_max=scale_max, scale_min=scale_min
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

    def save_as_image(self, output_path: Path, scale_max: float, scale_min: float):
        """
        Convert ScanImage data to an Image and save it to the given output_path.

        :param output_path: the given path to save the scan data.
        :return: the output path to where the image is saved.
        """
        self._to_pil_image(scale_max=scale_max, scale_min=scale_min).save(output_path)


def grayscale_to_rgba(scan_data: FloatArray2D) -> ImageRGBA:
    """
    Convert a 2D grayscale array to an 8-bit RGBA array.

    The grayscale pixel values are assumed to be floating point values in the [0, 255] interval.
    NaN values will be converted to black pixels with 100% transparency.

    :param scan_data: The grayscale image data to be converted to an 8-bit RGBA image.
    :returns: Array with the image data in 8-bit RGBA format.
    """
    gray_uint8 = np.nan_to_num(scan_data, nan=0.0).astype(np.uint8)
    rgba = np.repeat(gray_uint8[..., np.newaxis], 4, axis=-1)
    rgba[..., 3] = (~np.isnan(scan_data)).astype(np.uint8) * 255
    return rgba


def normalize_2d_array(
    array_to_normalize: FloatArray2D,
    scale_max: float = 255,
    scale_min: float = 25,
) -> FloatArray2D:
    """
    Normalize a 2D intensity map to a specified output range.
    The normalization is done by the steps:
    1. apply min-max normalization to grayscale data
    2. stretch / scale the normalized data from the unit range to a specified output range

    :note: If all valid pixels have the same value (no contrast), the output
    is filled with the midpoint of the output range. NaN pixels are preserved.

    :param array_to_normalize: 2D array of input intensity values.
    :param scale_max: Maximum output intensity value. Default is ``255``.
    :param scale_min: Minimum output intensity value. Default is ``25``.
    :returns: Normalized 2D intensity map with values in ``[scale_min, max_val]``.
    """
    imin = np.nanmin(array_to_normalize.data)
    imax = np.nanmax(array_to_normalize.data)

    if imax == imin:
        fill_value = (scale_min + scale_max) / 2
        result = np.full_like(array_to_normalize, fill_value)
        result[np.isnan(array_to_normalize)] = np.nan
        return result

    norm = (array_to_normalize - imin) / (imax - imin)
    return scale_min + (scale_max - scale_min) * norm
