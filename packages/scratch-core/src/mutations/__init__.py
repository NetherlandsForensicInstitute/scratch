"""
Image Mutations Module
======================

This package contains all available `ImageMutation` implementations.

Each mutation represents a single, well-defined transformation that can
be applied to a `ScanImage`. Mutations are designed to be composable and
can be chained together using a pipeline (e.g. `returns.pipeline.pipe`).
"""

from .filter import LevelMap
from .spatial import CropToMask, Resample


__all__ = ["LevelMap", "CropToMask", "Resample"]
