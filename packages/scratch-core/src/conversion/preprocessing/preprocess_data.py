"""
Orchestrate form and noise removal for surface preprocessing (Step 2).

FIXED to exactly match MATLAB PreprocessData.m lines 156-192.
"""

import numpy as np
from numpy.typing import NDArray

from conversion.preprocessing.preprocess_data_filter import (
    apply_gaussian_filter_1d,
    ALPHA_GAUSSIAN,
)


def _cheby_cutoff_to_gauss_sigma(cutoff: float, pixel_size: float) -> float:
    """
    Convert cutoff wavelength to Gaussian sigma using ISO 16610 standard.

    :param cutoff: Cutoff wavelength in physical units (e.g., meters).
    :param pixel_size: Pixel spacing in the same units as cutoff.
    :return: Gaussian sigma in pixel units.
    """
    cutoff_pixels = cutoff / pixel_size
    return cutoff_pixels * ALPHA_GAUSSIAN


def apply_form_noise_removal(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_hi: float,
    mask: NDArray[np.bool_] | None = None,
    cut_borders_after_smoothing: bool = True,
    cutoff_lo: float = 250e-6,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Apply form and noise removal (Step 2 of PreprocessData).

    This function implements lines 156-192 of PreprocessData.m, which perform
    the second step of surface preprocessing: removing large-scale form
    (curvature) and fine noise to isolate striation features.

    The algorithm has the following steps:

    **Step 1: Calculate sigma and check data size**
        Convert the cutoff wavelength to Gaussian sigma. If the data is
        too short (2*sigma > 20% of height), disable border cutting to
        preserve data.

    **Step 2: Shape removal**
        Use apply_gaussian_filter_1d with is_high_pass=True to remove
        large-scale form (curvature, tilt, waviness).

    **Step 3: Noise removal**
        Apply apply_gaussian_filter_1d with is_high_pass=False (lowpass)
        to remove high-frequency noise while preserving striation features.

    :param depth_data: 2D depth/height data array. Should already be coarsely aligned (Step 1 of PreprocessData). Rows
     perpendicular to striations.
    :param xdim: Pixel spacing in meters (m). Distance between adjacent measurements.
    :param cutoff_hi: High-frequency cutoff wavelength in meters (m) for shape removal. Larger values remove more form.
     Typical: 2000e-6 m (2000 um).
    :param mask: Boolean mask array (True = valid data). Masked regions are excluded from processing. Must match
     depth_data shape.
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts. May be automatically disabled for short
     data. Default True.
    :param cutoff_lo: Low-frequency cutoff wavelength in meters (m) for noise removal. Smaller values remove more noise.
     Default 250e-6 m (250 um).

    :returns depth_data: Processed data with form and noise removed. Contains the striation features isolated from
     background shape and fine noise.
    :returns mask: Boolean mask indicating valid data points after processing. True = valid data, False = invalid/masked
     region.

    Notes
    -----
    - This function replicates MATLAB PreprocessData.m lines 156-192
    - Short data (2*sigma > 20% height) automatically disables border cutting
    - The function assumes data has been coarsely aligned (Step 1 complete)
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)

    # -------------------------------------------------------------------------
    # Step 1: Calculate sigma and check if data is too short
    # -------------------------------------------------------------------------
    # Calculate Gaussian sigma from cutoff wavelength
    sigma = _cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)

    # Check if data is too short for border cutting
    # MATLAB line 163: if 2 * sigma > size(data_rot.depth_data, 1) * 0.2
    data_height = depth_data.shape[0]
    data_too_short = (2 * sigma) > (data_height * 0.2)

    # Override border cutting for short data (matches MATLAB lines 164-165)
    if data_too_short:
        effective_cut_borders = False
    else:
        effective_cut_borders = cut_borders_after_smoothing

    # -------------------------------------------------------------------------
    # Step 2: Shape removal (highpass filter)
    # -------------------------------------------------------------------------
    # MATLAB lines 172-174: RemoveShapeGaussian
    data_no_shape, _, mask_shape = apply_gaussian_filter_1d(
        depth_data,
        xdim=xdim,
        cutoff=cutoff_hi,
        is_high_pass=True,
        cut_borders_after_smoothing=effective_cut_borders,
        mask=mask,
    )

    # -------------------------------------------------------------------------
    # Step 3: Noise removal (lowpass filter)
    # -------------------------------------------------------------------------
    # MATLAB line 176: RemoveNoiseGaussian
    data_no_noise, _, mask_noise = apply_gaussian_filter_1d(
        data_no_shape,
        xdim=xdim,
        cutoff=cutoff_lo,
        is_high_pass=False,
        cut_borders_after_smoothing=effective_cut_borders,
        mask=mask_shape,
    )

    return data_no_noise, mask_noise
