from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

import numpy as np

from conversion.filters.array_manipulation import crop_nan_borders, pad_array
from conversion.filters.data_formats import FilterDomain
from conversion.filters.kernels import create_averaging_kernel, filter_kernel
from conversion.filters.validation import (
    _validate_filter_size,
    _validate_domain,
    _validate_regression_order,
)


class SurfaceFilter(ABC):
    """Abstract base class for surface filters."""

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

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to data.

        :param data: Input data array to be filtered.
        :return: Filtered data array.
        :raises ValueError: If data is invalid.
        """
        # Validate input
        data = np.asarray(data)
        if data.size == 0:
            raise ValueError("Input data array is empty")
        if data.ndim > 2:
            raise ValueError(f"Input data must be 1D or 2D, got {data.ndim}D")

        # Check if filtering is needed
        if not self._should_filter():
            return data

        # Crop empty (NaN) borders
        data_cropped, trim = crop_nan_borders(data)

        # Check if any valid data remains
        if data_cropped.size == 0 or np.all(np.isnan(data_cropped)):
            return data

        # Apply filtering
        filtered = self._filter_data(data_cropped)

        # Apply high-pass if requested
        # High-pass filtering extracts high-frequency components (roughness/texture)
        # by subtracting the low-frequency filtered result from the original data.
        # - Low-pass (is_high_pass=False): Returns smoothed data (form/waviness)
        # - High-pass (is_high_pass=True): Returns residual (roughness/fine details)
        # Common in surface metrology: roughness = profile - filtered_profile
        if self.is_high_pass:
            filtered = data_cropped - filtered

        # Restore original array size
        filtered = pad_array(filtered, trim.to_pad())

        return filtered

    @abstractmethod
    def _should_filter(self) -> bool:
        """Check if filtering should be applied."""
        raise NotImplementedError("Subclasses must implement _should_filter()")

    @abstractmethod
    def _filter_data(self, data: np.ndarray) -> np.ndarray:
        """Apply the actual filtering operation."""
        raise NotImplementedError("Subclasses must implement _filter_data()")


class KernelBasedFilter(SurfaceFilter):
    """Base class for filters that use kernel convolution."""

    # Type alias for kernel: either 2D array or tuple of (col, row) 1D arrays
    KernelType = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]

    def __init__(self, domain: FilterDomain, regression_order: int = 0, **kwargs):
        """Initialize kernel-based filter.

        :param domain: FilterDomain enum (DISK or RECTANGLE).
        :param regression_order: Polynomial regression order (0, 1, or 2).
        :param kwargs: Additional arguments passed to SurfaceFilter.
        """
        super().__init__(**kwargs)
        _validate_domain(domain, (FilterDomain.DISK, FilterDomain.RECTANGLE))
        self.domain = domain
        _validate_regression_order(regression_order)
        self.regression_order = regression_order
        self._kernel: KernelBasedFilter.KernelType
        self._generate_kernel()  # Subclass must implement

    @abstractmethod
    def _generate_kernel(self) -> None:
        """Generate filter kernel. Must be implemented by subclass.

        Subclasses must set self._kernel to either:
        - A 2D ndarray for non-separable kernels
        - A tuple (col_kernel, row_kernel) of 1D ndarrays for separable kernels
        """
        raise NotImplementedError("Subclasses must implement _generate_kernel()")

    def _unpack_kernel(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Unpack kernel into (kernel_col, kernel_row) format for filter_kernel."""
        if isinstance(self._kernel, tuple):
            return self._kernel[0], self._kernel[1]
        return self._kernel, None

    def _filter_data(self, data: np.ndarray) -> np.ndarray:
        """Apply kernel-based filtering."""
        kernel_col, kernel_row = self._unpack_kernel()
        return filter_kernel(
            kernel_col=kernel_col,
            kernel_row=kernel_row,
            data=data,
            regression_order=self.regression_order,
            nan_out=self.nan_out,
        )


class AveragingFilter(KernelBasedFilter):
    """Averaging (uniform) filter for surface data."""

    def __init__(
        self,
        filter_size: Tuple[int, int],
        domain: FilterDomain,
        regression_order: int = 0,
        **kwargs,
    ):
        """Initialize averaging filter.

        :param filter_size: Filter size in pixels (rows, cols).
        :param domain: FilterDomain enum - filter domain shape (DISK or RECTANGLE).
        :param regression_order: Polynomial regression order (0, 1, or 2).
        :param kwargs: Additional arguments passed to KernelBasedFilter.
        :raises ValueError: If parameters are invalid.
        """
        _validate_filter_size(filter_size)
        self.filter_size = np.asarray(filter_size, dtype=int).reshape(2)
        super().__init__(domain=domain, regression_order=regression_order, **kwargs)

    def _should_filter(self) -> bool:
        """Check if filter size is large enough."""
        return bool(np.any(self.filter_size > 1))

    def _generate_kernel(self):
        """Generate averaging filter kernel."""
        if self.domain == FilterDomain.RECTANGLE:
            # Separable 1D kernels: (col_kernel, row_kernel)
            kernel_col = np.ones((self.filter_size[0], 1)) / self.filter_size[0]
            kernel_row = np.ones((1, self.filter_size[1])) / self.filter_size[1]
            self._kernel = (kernel_col, kernel_row)
        else:
            # 2D disk kernel
            self._kernel = create_averaging_kernel(self.filter_size)

    def __repr__(self) -> str:
        """Return string representation of filter."""
        return (
            f"AveragingFilter(filter_size={tuple(self.filter_size)}, "
            f"domain=FilterDomain.{self.domain.name}, regression_order={self.regression_order}, "
            f"is_high_pass={self.is_high_pass})"
        )
