from abc import ABC, abstractmethod

import numpy as np

from conversion.filters.array_manipulation import crop_nan_borders, pad_array


class SurfaceFilter(ABC):
    """Abstract base class for surface filters."""

    def __init__(
        self,
        is_high_pass: bool = False,
        nan_out: bool = True,
        is_robust: bool = False,
        robust_conv_tol: float = 1e-2,
        n_robust_conv_iter: int = 50,
    ):
        """Initialize filter with common parameters.

        :param is_high_pass: Whether to apply as highpass filter (data - filtered).
        :param nan_out: Whether to return NaN values as NaN values.
        :param is_robust: Whether to use robust M-estimator (biweight).
        :param robust_conv_tol: Convergence tolerance for robust filtering.
        :param n_robust_conv_iter: Maximum iterations for robust filtering.
        :raises ValueError: If parameters are invalid.
        """
        if robust_conv_tol <= 0:
            raise ValueError(f"robust_conv_tol must be positive, got {robust_conv_tol}")
        if n_robust_conv_iter < 1:
            raise ValueError(
                f"n_robust_conv_iter must be >= 1, got {n_robust_conv_iter}"
            )

        self.is_high_pass = is_high_pass
        self.nan_out = nan_out
        self.is_robust = is_robust
        self.robust_conv_tol = robust_conv_tol
        self.n_robust_conv_iter = n_robust_conv_iter

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
