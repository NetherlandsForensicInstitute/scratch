from abc import ABC

import numpy as np

from conversion.filters.gaussian_filter import apply_gaussian_filter
from utils.array_definitions import ScanMap2DArray


class BaseFilter(ABC):
    def __init__(
        self,
        is_high_pass: bool = False,
        nan_out: bool = True,
    ):
        """Initialize filter with common parameters.
        :param is_high_pass: Whether to apply as highpass filter (data - filtered).
        :param nan_out: Whether to return NaN values as NaN values.
        :raises ValueError: If parameters are invalid.
        """
        self.is_high_pass = is_high_pass
        self.nan_out = nan_out

    def apply(self, data: ScanMap2DArray) -> ScanMap2DArray:
        """Apply filter to scan_image."""
        filtered = self._filter(data)

        if self.is_high_pass:
            filtered = data - filtered

        return filtered

    def _filter(self, data: ScanMap2DArray) -> ScanMap2DArray:
        """Apply the actual filtering operation."""
        raise NotImplementedError("Subclasses must implement _filter_data()")


class GaussianFilter(BaseFilter):
    """Gaussian filter for surface data."""

    def __init__(
        self,
        cutoff_length: tuple[float, float],
        pixel_size: tuple[float, float] = (1.0, 1.0),
        is_high_pass: bool = False,
        nan_out: bool = True,
    ):
        """Initialize Gaussian filter.

        :param cutoff_length: Cutoff wavelength (row, col) in physical units.
        :param pixel_size: The pixel size in the X-direction en Y-direction in meters (m).
        :param is_high_pass: If True, return data - filtered (high-pass).
        :param nan_out: If True, preserve NaN positions in output.
        """
        self.cutoff_length = cutoff_length
        self.pixel_size = pixel_size
        super().__init__(is_high_pass, nan_out)

    def _filter(self, data: ScanMap2DArray) -> ScanMap2DArray:
        """Apply the actual filtering operation."""
        if np.all(np.isnan(self.cutoff_length)):
            return data

        return apply_gaussian_filter(
            data,
            cutoff_length=self.cutoff_length,
            pixel_size=self.pixel_size,
            nan_out=self.nan_out,
        )
