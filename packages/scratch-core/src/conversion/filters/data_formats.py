from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import numpy as np


class FilterDomain(Enum):
    """Filter domain shape enumeration."""

    DISK = auto()
    RECTANGLE = auto()


@dataclass
class Trim1D:
    """Trim amounts for 1D array borders."""

    start: int
    end: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.start, self.end])

    def to_pad(self) -> "Pad1D":
        """Convert to Pad1D object."""
        return Pad1D(start=self.start, end=self.end)


@dataclass
class Trim2D:
    """Trim amounts for 2D array borders."""

    top: int
    bottom: int
    left: int
    right: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.top, self.bottom, self.left, self.right])

    def to_pad(self) -> "Pad2D":
        """Convert to Pad2D object."""
        return Pad2D(top=self.top, bottom=self.bottom, left=self.left, right=self.right)


@dataclass
class Pad1D:
    """Pad amounts for 1D array borders."""

    start: int
    end: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.start, self.end])


@dataclass
class Pad2D:
    """Pad amounts for 2D array borders."""

    top: int
    bottom: int
    left: int
    right: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.top, self.bottom, self.left, self.right])


TrimType = Union[Trim1D, Trim2D]
PadType = Union[Pad1D, Pad2D]
