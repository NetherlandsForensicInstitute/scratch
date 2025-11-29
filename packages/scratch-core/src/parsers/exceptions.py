class ExportError(Exception):
    """Raised when an error occurs during export."""

    def __init__(self, message: str):
        super().__init__(message)
