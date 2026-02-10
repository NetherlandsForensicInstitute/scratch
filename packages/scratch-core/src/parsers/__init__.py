"""File parsing and serialization.

This module provides functions for parsing scan image data in x3p.

File Format Support
-------------------
**Input Formats**:

- AL3D: Alicona 3D surface files (via :meth:`~container_models.image.ImageContainer.from_scan_file`)
- X3P: ISO 25178-72 XML format for surface texture data

**Output Formats**:

- X3P: ISO 25178-72 XML format (unit converted to meters)

.. note::

    All spatial measurements are standardized to meters (m).
    Custom formats can be added via surfalize FileHandler registration.
"""
