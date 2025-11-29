class ImageGenerationError(Exception):
    """Raised when an error occurs during image generation."""

    def __init__(self, message: str):
        super().__init__(message)
