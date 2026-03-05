class ArrayShapeMismatchError(Exception):
    def __init__(self, size: int, target_shape: tuple[int, ...]):
        self.size = size
        self.target_shape = target_shape

    def __str__(self) -> str:
        return f"Cannot reshape array of size {self.size} to shape {self.target_shape}."
