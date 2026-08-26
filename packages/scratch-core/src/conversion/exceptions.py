class NoValidGridCellsError(Exception):
    """Raised when no valid grid cells are generated."""

    def __str__(self) -> str:
        return "No valid grid cells are generated."
