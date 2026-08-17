class ImageNotIsotropicError(Exception):
    """Raised when an image is not isotropic."""

    def __init__(self, scale_x: float | int, scale_y: float | int):
        super().__init__(scale_x, scale_y)
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __str__(self) -> str:
        return f"Image is not isotropic: scale_x={self.scale_x}, while scale_y={self.scale_y}."


class NoValidGridCellsError(Exception):
    def __str__(self) -> str:
        return "No valid grid are cells generated."
