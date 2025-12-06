from enum import Flag, auto
from typing import Protocol

from utils.array_definitions import ScanMap2DArray


class FilterFlags(Flag):
    NAN_OUT = auto()
    HIGH_PASS = auto()


class Filter(Protocol):
    def __call__(
        self,
        data: ScanMap2DArray,
        alpha: float,
        *,
        cutoff_pixels: tuple[float, float],
        flags: FilterFlags,
    ) -> ScanMap2DArray: ...
