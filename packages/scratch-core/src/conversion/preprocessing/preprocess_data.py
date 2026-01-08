"""
Orchestrate form and noise removal for surface preprocessing (Step 2). It coordinates shape removal and noise removal
 with optional bullet unfolding for slope correction.

The pipeline handles:
    1. Calculating Gaussian sigma from cutoff wavelength
    2. Checking if data is too short for border cutting
    3. Applying shape removal (or UnfoldBullet for slope correction)
    4. Applying noise removal
"""

import numpy as np
from numpy.typing import NDArray

from conversion.preprocessing.cheby_cutoff_to_gauss_sigma import (
    cheby_cutoff_to_gauss_sigma,
)
from conversion.preprocessing.remove_noise_gaussian import remove_noise_gaussian
from conversion.preprocessing.remove_shape_gaussian import remove_shape_gaussian
from conversion.preprocessing.unfold_bullet import unfold_bullet


def apply_form_noise_removal(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_hi: float,
    mask: NDArray[np.bool_] | None = None,
    slope_correction: bool = False,
    cut_borders_after_smoothing: bool = True,
    cutoff_lo: float = 250e-6,
) -> tuple[NDArray[np.floating], NDArray[np.bool_], float | None]:
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
        Either:
        - If slope_correction=True: Use UnfoldBullet to correct for
          bullet surface curvature while extracting striations
        - If slope_correction=False: Use RemoveShapeGaussian to apply
          highpass filtering and remove large-scale form

    **Step 3: Noise removal**
        Apply RemoveNoiseGaussian lowpass filter to remove high-frequency
        noise while preserving striation features.

    Parameters
    ----------
    depth_data : NDArray[np.floating]
        2D depth/height data array. Should already be coarsely aligned
        (Step 1 of PreprocessData). Rows perpendicular to striations.
    xdim : float
        Pixel spacing in meters (m). Distance between adjacent measurements.
    cutoff_hi : float
        High-frequency cutoff wavelength in meters (m) for shape removal.
        Larger values remove more form. Typical: 2000e-6 m (2000 um).
    mask : NDArray[np.bool_] | None, optional
        Boolean mask array (True = valid data). Masked regions are excluded
        from processing. Must match depth_data shape.
    slope_correction : bool, optional
        If True, use UnfoldBullet to correct for bullet surface curvature.
        If False, use standard RemoveShapeGaussian. Default False.
    cut_borders_after_smoothing : bool, optional
        If True, crop filter edge artifacts. May be automatically disabled
        for short data. Default True.
    cutoff_lo : float, optional
        Low-frequency cutoff wavelength in meters (m) for noise removal.
        Smaller values remove more noise. Default 250e-6 m (250 um).

    Returns
    -------
    depth_data : NDArray[np.floating]
        Processed data with form and noise removed. Contains the striation
        features isolated from background shape and fine noise.
    mask : NDArray[np.bool_]
        Boolean mask indicating valid data points after processing.
        True = valid data, False = invalid/masked region.
    relative_highest_point_location : float | None
        For bullet data with slope_correction=True, the relative position
        (0-1) of the highest point on the bullet surface. None if
        slope_correction is False.

    Example
    -------
    >>> import numpy as np
    >>> rows = np.linspace(0, 100, 500)
    >>> shape = 0.001 * rows**2  # Parabolic form
    >>> striations = 0.01 * np.sin(2 * np.pi * rows / 10)
    >>> noise = np.random.randn(500) * 0.001
    >>> surface = np.tile((shape + striations + noise).reshape(-1, 1), (1, 100))
    >>> depth_data, mask, highest_point = apply_form_noise_removal(
    ...     surface, xdim=1e-6, cutoff_hi=2000e-6
    ... )

    Notes
    -----
    - This function replicates MATLAB PreprocessData.m lines 156-192
    - Short data (2*sigma > 20% height) automatically disables border cutting
    - For bullet data, use slope_correction=True to correct for curvature
    - The function assumes data has been coarsely aligned (Step 1 complete)
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)

    # Initialize output variables
    relative_highest_point_location: float | None = None

    # -------------------------------------------------------------------------
    # Step 1: Calculate sigma and check if data is too short
    # -------------------------------------------------------------------------
    # Calculate Gaussian sigma from cutoff wavelength
    sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)

    # Check if data is too short for border cutting
    # MATLAB: if 2 * sigma > size(data_rot.depth_data, 1) * 0.2
    data_height = depth_data.shape[0]
    data_too_short = (2 * sigma) > (data_height * 0.2)

    # Override border cutting for short data
    if data_too_short:
        effective_cut_borders = False
    else:
        effective_cut_borders = cut_borders_after_smoothing

    # -------------------------------------------------------------------------
    # Step 2: Shape removal (or bullet unfolding)
    # -------------------------------------------------------------------------
    if slope_correction:
        # Use UnfoldBullet for bullet data with slope correction
        data_no_shape, _, unfold_mask, relative_highest_point_location = unfold_bullet(
            depth_data,
            xdim=xdim,
            cutoff_hi=cutoff_hi,
            cutoff_lo=cutoff_lo,
            cut_borders_after_smoothing=effective_cut_borders,
            mask=mask,
        )

        # Handle None mask from unfolding
        mask_shape = (
            unfold_mask
            if unfold_mask is not None
            else np.ones(data_no_shape.shape, dtype=bool)
        )

    else:
        # Use standard shape removal for toolmarks
        data_no_shape, _, mask_shape = remove_shape_gaussian(
            depth_data,
            xdim=xdim,
            cutoff_hi=cutoff_hi,
            cut_borders_after_smoothing=effective_cut_borders,
            mask=mask,
        )

    # -------------------------------------------------------------------------
    # Step 3: Noise removal
    # -------------------------------------------------------------------------
    data_no_noise, _, mask_noise = remove_noise_gaussian(
        data_no_shape,
        xdim=xdim,
        cutoff_lo=cutoff_lo,
        cut_borders_after_smoothing=effective_cut_borders,
        mask=mask_shape,
    )

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------
    return data_no_noise, mask_noise, relative_highest_point_location
