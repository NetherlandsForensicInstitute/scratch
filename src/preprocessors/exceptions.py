class ArrayShapeMismatchError(Exception):
    def __init__(self, size: int, target_shape: tuple[int, ...]):
        super().__init__(size, target_shape)
        self.size = size
        self.target_shape = target_shape

    def __str__(self) -> str:
        return f"Cannot reshape array of size {self.size} to shape {self.target_shape}."


class MaskFullError(Exception):
    """Raised when a mask covers the entire scan image."""

    def __str__(self) -> str:
        return "Mask covers the entire scan image: no data remains after masking."
