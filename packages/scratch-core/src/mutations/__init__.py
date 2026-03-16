"""
Image Mutations Module
======================

This package contains all available `ImageMutation` implementations.

Each mutation represents a single, well-defined transformation that can
be applied to a `ScanImage`. Mutations are designed to be composable and
can be chained together using a pipeline (e.g. `returns.pipeline.pipe`).
"""

from .filter import GaussianRegressionFilter, LevelMap, Mask
from .spatial import CropToMask, Resample, Rotate

__all__ = [
    "LevelMap",
    "GaussianRegressionFilter",
    "Mask",
    "Resample",
    "CropToMask",
    "Rotate",
]
