"""
File parsing and serialization utilities for scan image data.

This module provides functions for loading, parsing, and saving scan image data
in various file formats, with automatic conversion to and from the internal
ScanImage container model. All parsers are designed to work within railway-oriented
programming pipelines, returning Result/IOResult containers for safe error handling.

The module handles two primary workflows:
1. **Loading**: Parse external file formats into ScanImage containers
2. **Saving**: Convert ScanImage containers to external file formats

File Format Support
-------------------
**Input Formats** (via load_scan_image):
- AL3D: Alicona 3D surface files (with custom patch)
- X3P: ISO 25178-72 XML format for surface texture data
- Automatic unit conversion to meters (SI base unit)
- Optional subsampling via step_size parameters

**Output Formats**:
- X3P: ISO 25178-72 XML format for surface texture data
- Configurable metadata via X3PMetaData

Railway Integration
-------------------
All parser functions return Result or IOResult containers and are decorated
with logging functionality. This enables seamless integration into functional
pipelines with automatic error propagation:

Notes
-----
- All spatial measurements are standardized to meters (m) for consistency
- Parsers use railway-oriented programming patterns for robust error handling
- Custom file format support can be added via surfalize FileHandler registration
"""

from .loaders import load_scan_image
from .x3p import X3PMetaData, save_x3p, parse_to_x3p

__all__ = (
    "load_scan_image",
    "parse_to_x3p",
    "save_x3p",
    "X3PMetaData",
)
