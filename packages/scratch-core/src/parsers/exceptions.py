from enum import StrEnum


class ExportError(Exception):
    """Raised when an error occurs during export."""

    def __init__(self, message: str):
        super().__init__(message)


class PreProcessError(StrEnum):
    X3P_PARSE_ERROR = "Failed to parse image X3P"
    X3P_WRITE_ERROR = "Failed to write X3P file"
    SURFACE_LOAD_ERROR = "Failed to load image file"
