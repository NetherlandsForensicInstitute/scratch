"""Image container architecture.

This module defines the data container used to represent images throughout
the processing pipeline.

Architecture
------------
::

    +--------------------------------------+
    |           ImageContainer             |
    |--------------------------------------|
    | data     : DepthData                 |
    | metadata : MetaData                  |
    | height   : int (rows)                |
    | width    : int (columns)             |
    +--------------------------------------+
    | from_scan_file(path) -> cls          |
    | valid_mask -> BinaryMask             |
    | valid_data -> BoolArray1D            |
    | center -> Coordinate                 |
    | rgba -> ImageRGBA                    |
    | export_png(path) -> Path             |
    | export_x3p(path) -> Path             |
    +--------------------------------------+

- :class:`ImageContainer` is the single container for all image data.
- Stores depth data as a 2D float64 NumPy array with scale metadata.
- Compared by data equality (NaN-aware).

.. note::

    Mutations and computations should operate on :class:`ImageContainer`
    rather than on raw NumPy arrays where possible.
"""

from __future__ import annotations
from functools import lru_cache
from surfalize.file import FileHandler
from pathlib import Path
from sre_constants import MAGIC
import numpy as np
from parsers.patches.al3d import read_al3d
from parsers.x3p import parse_to_x3p
from pydantic import BaseModel, ConfigDict
from scipy.constants import femto, micro
from PIL.Image import fromarray
from surfalize import Surface

from container_models.base import (
    BinaryMask,
    BoolArray1D,
    Coordinate,
    DepthData,
    Pair,
    Scale,
    ImageRGBA,
)


class MetaData(BaseModel):
    scale: Scale

    @property
    def is_isotropic(self) -> bool:
        return bool(np.isclose(*tuple(self.scale), atol=femto))

    @property
    def central_diff_scales(self) -> Scale:
        return self.scale / 2

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        regex_engine="rust-regex",
    )


class ImageContainer(BaseModel):
    data: DepthData
    metadata: MetaData

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        regex_engine="rust-regex",
        revalidate_instances="always",
    )

    @property
    def height(self) -> int:
        """Return the height (number of rows) of the image."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Return the width (number of columns) of the image."""
        return self.data.shape[1]

    @property
    def center(self) -> Coordinate:
        return self.metadata.central_diff_scales.map(
            lambda x, y: (x - 1) * y, other=reversed(self.data.shape)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageContainer):
            return NotImplemented
        return np.array_equal(self.data, other.data, equal_nan=True)

    @property
    def valid_mask(self) -> BinaryMask:
        """Mask of the valid pixels in the data."""
        valid_mask = ~np.isnan(self.data)
        valid_mask.setflags(write=False)
        return valid_mask

    @property
    def valid_data(self) -> BoolArray1D:
        """Valid pixels in the data."""
        valid_data = self.data[self.valid_mask].astype(np.bool_)
        valid_data.setflags(write=False)
        return valid_data

    @classmethod
    @lru_cache(maxsize=1)
    def from_scan_file(cls, scan_file: Path) -> ImageContainer:
        """
        Load a scan image from a file. Parsed values will be converted to meters (m).
        :param scan_file: The path to the file containing the scanned image data.
        :returns: An instance of `ImageContainer`.
        """
        FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)
        surface = Surface.load(scan_file)

        return cls(
            data=np.asarray(surface.data, dtype=np.float64) * micro,
            metadata=MetaData(
                scale=Pair(surface.step_x, surface.step_y) * micro,
            ),
        )

    @property
    def rgba(self) -> ImageRGBA:
        gray_uint8 = np.nan_to_num(self.data, nan=0.0).astype(np.uint8)
        rgba = np.repeat(gray_uint8[..., np.newaxis], 4, axis=-1)
        rgba[..., 3] = (~np.isnan(self.data)).astype(np.uint8) * 255
        return rgba

    def export_png(self, output_path: Path) -> Path:
        """
        Save an image to disk.
        :param output_path: The path where the image should be written.
        :returns: The path to the saved image.
        """
        fromarray(self.rgba).save(output_path)
        return output_path

    def export_x3p(self, output_path: Path) -> Path:
        """
        Save an X3P file to disk.
        :param output_path: The path where the file should be written.
        :returns: The path to the saved file.
        """
        parse_to_x3p(self).unwrap().write(str(output_path))
        return output_path
