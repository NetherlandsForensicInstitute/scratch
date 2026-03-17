"""
Rendering and visualization utilities for 3D surface scan data.

This module provides functions for computing surface properties, applying lighting
models, and generating visual representations of 3D scan data. All functions are
designed for railway-oriented programming pipelines,

Notes
-----
- Surface normals are computed using central differences with NaN padding at borders
- NaN values in scan data are handled gracefully throughout the pipeline
- All lighting calculations preserve physical units and scale information
- Output images use RGBA format with transparency for invalid (NaN) regions
"""

from .image_io import (
    get_scan_image_for_display,
    grayscale_to_image,
    grayscale_to_rgba,
    save_image,
    scan_to_image,
)
from .normalizations import compute_surface_normals, normalize_2d_array
from .shading import apply_multiple_lights

__all__ = (
    "apply_multiple_lights",
    "compute_surface_normals",
    "get_scan_image_for_display",
    "normalize_2d_array",
    "save_image",
    "grayscale_to_rgba",
    "grayscale_to_image",
    "scan_to_image",
)
