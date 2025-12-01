import numpy as np
from scipy import ndimage

_GAUSSIAN_ALPHA = np.sqrt(np.log(2) / np.pi)


def _cutoff_to_sigma(cutoff_length: float) -> float:
    """Convert MATLAB-style cutoff length to scipy sigma.

    MATLAB Gaussian: exp(-pi * (x / (alpha * cutoff))^2)
    scipy Gaussian:  exp(-x^2 / (2 * sigma^2))

    Matching: sigma = alpha * cutoff / sqrt(2 * pi)

    :param cutoff_length: Cutoff wavelength in pixels.
    :return: Equivalent scipy sigma.
    """
    return _GAUSSIAN_ALPHA * cutoff_length / np.sqrt(2 * np.pi)


def _cutoff_to_truncate(cutoff_length: float, sigma: float) -> float:
    """Calculate scipy truncate parameter to match MATLAB kernel size.

    MATLAB kernel radius = ceil(cutoff)
    scipy kernel radius = ceil(truncate * sigma)

    :param cutoff_length: Cutoff wavelength in pixels.
    :param sigma: scipy sigma parameter.
    :return: truncate parameter for scipy.
    """
    return np.ceil(cutoff_length) / sigma


def gaussian_filter(
    data: np.ndarray,
    cutoff_length: tuple[float, float],
    pixel_separation: tuple[float, float] = (1.0, 1.0),
    nan_out: bool = True,
) -> np.ndarray:
    """Apply Gaussian filter to 2D data with NaN handling.

    :param data: Input 2D data array.
    :param cutoff_length: Cutoff wavelength (row, col) in physical units.
    :param pixel_separation: Pixel separation (row, col) in physical units.
    :param nan_out: If True, preserve NaN positions in output.
    :return: Filtered data array.
    """
    data = np.asarray(data, dtype=float)

    # Convert cutoff to pixel units and scipy parameters
    cutoff_pixels = np.array(cutoff_length) / np.array(pixel_separation)
    sigma = np.array([_cutoff_to_sigma(c) for c in cutoff_pixels])
    truncate = max(_cutoff_to_truncate(c, s) for c, s in zip(cutoff_pixels, sigma))

    # Weighted filtering for NaN handling
    weights = (~np.isnan(data)).astype(float)
    data_clean = np.where(np.isnan(data), 0, data)

    filtered = ndimage.gaussian_filter(
        data_clean, sigma=sigma, mode="constant", cval=0.0, truncate=truncate
    )
    weight_sum = ndimage.gaussian_filter(
        weights, sigma=sigma, mode="constant", cval=0.0, truncate=truncate
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        result = filtered / weight_sum

    if nan_out:
        result[np.isnan(data)] = np.nan

    return result


class GaussianFilter:
    """Gaussian filter for surface data."""

    def __init__(
        self,
        cutoff_length: tuple[float, float],
        pixel_separation: tuple[float, float] = (1.0, 1.0),
        is_high_pass: bool = False,
        nan_out: bool = True,
    ):
        """Initialize Gaussian filter.

        :param cutoff_length: Cutoff wavelength (row, col) in physical units.
        :param pixel_separation: Pixel separation (row, col) in physical units.
        :param is_high_pass: If True, return data - filtered (high-pass).
        :param nan_out: If True, preserve NaN positions in output.
        """
        self.cutoff_length = tuple(cutoff_length)
        self.pixel_separation = tuple(pixel_separation)
        self.is_high_pass = is_high_pass
        self.nan_out = nan_out

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to data.

        :param data: Input 2D data array.
        :return: Filtered data array.
        """
        data = np.asarray(data, dtype=float)

        if data.size == 0:
            raise ValueError("Input data is empty")
        if data.ndim != 2:
            raise ValueError(f"Input must be 2D, got {data.ndim}D")

        # Skip if no valid cutoff
        if all(np.isnan(c) for c in self.cutoff_length):
            return data

        filtered = gaussian_filter(
            data,
            cutoff_length=self.cutoff_length,
            pixel_separation=self.pixel_separation,
            nan_out=self.nan_out,
        )

        if self.is_high_pass:
            filtered = data - filtered

        return filtered

    def __repr__(self) -> str:
        return (
            f"GaussianFilter(cutoff_length={self.cutoff_length}, "
            f"pixel_separation={self.pixel_separation}, "
            f"is_high_pass={self.is_high_pass})"
        )
