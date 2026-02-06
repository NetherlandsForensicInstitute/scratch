"""File parsing and serialization.

This module provides functions for loading, parsing, and saving scan image data
in various file formats, with automatic conversion to and from the internal
:class:`~container_models.image.ProcessImage` container model.

Workflows
---------
1. **Loading**: Parse external file formats into ProcessImage containers
2. **Saving**: Convert ProcessImage containers to external file formats

File Format Support
-------------------
**Input Formats**:

- AL3D: Alicona 3D surface files (via :meth:`ProcessImage.from_scan_file`)
- X3P: ISO 25178-72 XML format for surface texture data

**Output Formats**:

- X3P: ISO 25178-72 XML format (unit converted to meters)
- PNG: Preview images via :meth:`ProcessImage.export_png`

.. note::

    All spatial measurements are standardized to meters (m).
    Custom formats can be added via surfalize FileHandler registration.
"""
