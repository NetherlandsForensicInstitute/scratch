"""Noise/shape removal by applying low/high pass Gaussian filter."""

# TODO streamline with zero'th order filter in filter.py.
# What we know is that the code below uses a 1D Gaussian local filter while the code in filter.py uses a 2D Gaussian
# local filter.

import numpy as np
from numpy.typing import NDArray

from conversion.preprocessing.preprocess_data_filter import (
    apply_gaussian_filter_1d,
    cheby_cutoff_to_gauss_sigma,
)


def apply_shape_noise_removal(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_hi: float,
    mask: NDArray[np.bool_] | None = None,
    cut_borders_after_smoothing: bool = True,
    cutoff_lo: float = 250e-6,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Apply large-scale shape and noise removal to isolate striation features.

    The function has the following steps:

    **Step 1: Calculate sigma and check data size**
        Convert the cutoff wavelength to Gaussian sigma. If the data is
        too short (2*sigma > 20% of height), disable border cutting to
        preserve data.

    **Step 2: Shape removal**
        Use apply_gaussian_filter_1d with is_high_pass=True to remove
        large-scale shape (curvature, tilt, waviness).

    **Step 3: Noise removal**
        Apply apply_gaussian_filter_1d with is_high_pass=False (lowpass)
        to remove high-frequency noise while preserving striation features.


    :param depth_data: 2D depth/height data array. Should already be coarsely aligned. Rows perpendicular to striations.
    :param xdim: Pixel spacing in meters (m). Distance between adjacent measurements.
    :param cutoff_hi: High-frequency cutoff wavelength in meters (m) for shape removal. Larger values remove more global
     effects.
     Typical: 2000e-6 m (2000 um).
    :param mask: Boolean mask array (True = valid data). Masked regions are excluded from processing. Must match
     depth_data shape.
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts. May be automatically disabled for short
     data. Default True.
    :param cutoff_lo: Low-frequency cutoff wavelength in meters (m) for noise removal. Smaller values remove more noise.
     Default 250e-6 m (250 um).

    :returns depth_data: Processed data with shape and noise removed. Should contain the striation features isolated
    from background shape and fine noise.
    :returns mask: Boolean mask indicating valid data points after processing. True = valid data, False = invalid/masked
     region.

    Notes
    -----
    - Short data (2*sigma > 20% height) automatically disables border cutting.
    - The function assumes data has been coarsely aligned with striations from top to bottom.
    - Filters are applied perpendicular to striations (so row-wise).
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)

    # -------------------------------------------------------------------------
    # Step 1: Calculate sigma and check if data is too short
    # -------------------------------------------------------------------------
    # Calculate Gaussian sigma from cutoff wavelength
    sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)

    # Check if data is too short for border cutting
    data_height = depth_data.shape[0]
    data_too_short = (2 * sigma) > (data_height * 0.2)

    # Override border cutting for short data
    if data_too_short:
        cut_borders = False
    else:
        cut_borders = cut_borders_after_smoothing

    # -------------------------------------------------------------------------
    # Step 2: Shape removal (highpass filter)
    # -------------------------------------------------------------------------
    data_no_shape, _, mask_shape = apply_gaussian_filter_1d(
        depth_data,
        xdim=xdim,
        cutoff=cutoff_hi,
        is_high_pass=True,
        cut_borders_after_smoothing=cut_borders,
        mask=mask,
    )

    # -------------------------------------------------------------------------
    # Step 3: Noise removal (lowpass filter)
    # -------------------------------------------------------------------------
    data_no_noise, _, mask_noise = apply_gaussian_filter_1d(
        data_no_shape,
        xdim=xdim,
        cutoff=cutoff_lo,
        is_high_pass=False,
        cut_borders_after_smoothing=cut_borders,
        mask=mask_shape,
    )

    return data_no_noise, mask_noise
