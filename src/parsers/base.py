import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image
from surfalize import Surface

from src.parsers.data_types import ParsedImage


class Parser(ABC):
    @abstractmethod
    def parse(self, filepath: os.PathLike | Path) -> ParsedImage:
        """Parse surface data from a file."""
        raise NotImplementedError


class ScanFileParser(Parser):
    def parse(self, filepath: os.PathLike | Path) -> ParsedImage:
        """Parse surface data from a scan file."""
        surface = Surface.load(filepath)
        surface.show()
        return ParsedImage(
            data=surface.data,
            scale_x=surface.step_x,
            scale_y=surface.step_y,
            metadata=surface.metadata,
        )


class ImageFileParser(Parser):
    def parse(self, filepath: os.PathLike | Path) -> ParsedImage:
        """Parse surface data from an image file."""
        image = Image.open(filepath).convert("L")  # convert parsed image to grayscale
        data = np.array(image, dtype=np.float64) / 255.0  # scale pixel values to unit range
        return ParsedImage(data=data)
