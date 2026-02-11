"""
Rendering and visualization utilities for 3D surface scan data.

This module provides functions for computing surface properties, applying lighting
models, and generating visual representations of 3D scan data. All functions are
designed for railway-oriented programming pipelines, returning Result/IOResult
containers for safe error handling.

Notes
-----
- Surface normals are computed using central differences with NaN padding at borders
- NaN values in scan data are handled gracefully throughout the pipeline
- All lighting calculations preserve physical units and scale information
- Output images use RGBA format with transparency for invalid (NaN) regions
"""

from .normalizations import normalize_2d_array, normalize_to_surface_normals
from .shading import combine_light_components

__all__ = (
    "combine_light_components",
    "normalize_2d_array",
    "normalize_to_surface_normals",
)
