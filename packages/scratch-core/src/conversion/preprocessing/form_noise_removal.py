"""Orchestrate form and noise removal for surface preprocessing (Step 2).

This module implements the form and noise removal pipeline from PreprocessData.m
lines 156-192. It coordinates shape removal and noise removal with optional
bullet unfolding for slope correction.

The pipeline handles:
    1. Calculating Gaussian sigma from cutoff wavelength
    2. Checking if data is too short for border cutting
    3. Applying shape removal (or UnfoldBullet for slope correction)
    4. Applying noise removal

Migrated from MATLAB: PreprocessData.m lines 156-192
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from conversion.cheby_cutoff_to_gauss_sigma import cheby_cutoff_to_gauss_sigma
from conversion.remove_noise_gaussian import remove_noise_gaussian
from conversion.remove_shape_gaussian import remove_shape_gaussian
from conversion.unfold_bullet import unfold_bullet


@dataclass
class FormNoiseRemovalResult:
    """Result container for form and noise removal operation.

    This is the output of Step 2 (Form and noise removal) in the
    PreprocessData pipeline.

    Attributes:
        depth_data: The processed depth data with form and noise removed.
            Contains the striation features isolated from background shape
            and fine noise.
        mask: Boolean mask indicating valid data points after processing.
            True = valid data, False = invalid/masked region.
        relative_highest_point_location: For bullet data with slope_correction,
            this indicates the relative position (0-1) of the highest point
            on the bullet surface. None if slope_correction is False.
    """

    depth_data: NDArray[np.floating]
    mask: NDArray[np.bool_]
    relative_highest_point_location: float | None


def apply_form_noise_removal(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_hi: float,
    mask: NDArray[np.bool_] | None = None,
    slope_correction: bool = False,
    cut_borders_after_smoothing: bool = True,
    cutoff_lo: float = 250.0,
) -> FormNoiseRemovalResult:
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

    The MATLAB code structure being replicated::

        %% Step 2: Form and noise removal
        mask_noise = mask_align;

        if shape_noise_removal
            sigma = ChebyCutoffToGaussSigma(cutoff_hi, xdim);

            % For short pieces of data, override border cutting
            if 2 * sigma > size(data_rot.depth_data, 1) * 0.2
                param_tmp = param;
                param_tmp.cut_borders_after_smoothing = 0;

                if slope_correction
                    [data_rot_no_shape, ~, mask_shape, relative_highest_point_location] = ...
                        UnfoldBullet(data_rot, param_tmp, mask_align);
                else
                    [data_rot_no_shape, ~, mask_shape] = RemoveShapeGaussian(...);
                end

                [data_rot_no_noise, ~, mask_noise] = RemoveNoiseGaussian(...);
            else
                % Same but with original params (border cutting enabled)
                ...
            end
        end

    :param depth_data: 2D depth/height data array. Should already be coarsely
        aligned (Step 1 of PreprocessData). Rows perpendicular to striations.
    :param xdim: Pixel spacing in meters (m). Distance between adjacent
        measurements in the scan.
    :param cutoff_hi: High-frequency cutoff wavelength in micrometers (um)
        for shape removal. Larger values remove more form. Typical: 2000 um.
    :param mask: Optional boolean mask array (True = valid data). Masked
        regions are excluded from processing. Must match depth_data shape.
    :param slope_correction: If True, use UnfoldBullet to correct for bullet
        surface curvature. If False, use standard RemoveShapeGaussian.
        Default False (for toolmarks and general use).
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts.
        May be automatically disabled for short data. Default True.
    :param cutoff_lo: Low-frequency cutoff wavelength in micrometers (um)
        for noise removal. Smaller values remove more noise. Default 250 um.
    :return: FormNoiseRemovalResult containing:
        - depth_data: Processed data with form and noise removed
        - mask: Output mask after processing
        - relative_highest_point_location: Bullet highest point (if slope_correction)

    Example:
        >>> import numpy as np
        >>> # Create synthetic surface with shape + striations + noise
        >>> rows = np.linspace(0, 100, 500)
        >>> shape = 0.001 * rows**2  # Parabolic form
        >>> striations = 0.01 * np.sin(2 * np.pi * rows / 10)  # Periodic striations
        >>> noise = np.random.randn(500) * 0.001  # Fine noise
        >>> surface = np.tile((shape + striations + noise).reshape(-1, 1), (1, 100))
        >>> # Apply form and noise removal
        >>> result = apply_form_noise_removal(
        ...     surface, xdim=1e-6, cutoff_hi=2000, cutoff_lo=250
        ... )
        >>> # result.depth_data now contains primarily the striation pattern

    Example with bullet slope correction:
        >>> # For bullet data, enable slope correction
        >>> result = apply_form_noise_removal(
        ...     bullet_surface, xdim=1e-6, cutoff_hi=2000,
        ...     slope_correction=True
        ... )
        >>> print(f"Bullet highest point: {result.relative_highest_point_location:.2%}")

    Note:
        - This function replicates MATLAB PreprocessData.m lines 156-192
        - Short data (2*sigma > 20% height) automatically disables border cutting
        - For bullet data, use slope_correction=True to correct for curvature
        - The function assumes data has been coarsely aligned (Step 1 complete)
    """
    # Ensure 2D input
    depth_data = np.atleast_2d(depth_data)
    if depth_data.shape[0] == 1:
        depth_data = depth_data.T

    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)
    else:
        mask = np.atleast_2d(mask)
        if mask.shape[0] == 1:
            mask = mask.T

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
        unfold_result = unfold_bullet(
            depth_data,
            xdim=xdim,
            cutoff_hi=cutoff_hi,
            cutoff_lo=cutoff_lo,
            cut_borders_after_smoothing=effective_cut_borders,
            mask=mask,
        )

        # Extract results from unfolding
        data_no_shape = unfold_result.depth_data
        mask_shape = (
            unfold_result.mask
            if unfold_result.mask is not None
            else np.ones(data_no_shape.shape, dtype=bool)
        )
        relative_highest_point_location = unfold_result.relative_highest_point_location

    else:
        # Use standard shape removal for toolmarks
        shape_result = remove_shape_gaussian(
            depth_data,
            xdim=xdim,
            cutoff_hi=cutoff_hi,
            cut_borders_after_smoothing=effective_cut_borders,
            mask=mask,
        )

        data_no_shape = shape_result.depth_data
        mask_shape = shape_result.mask

    # -------------------------------------------------------------------------
    # Step 3: Noise removal
    # -------------------------------------------------------------------------
    noise_result = remove_noise_gaussian(
        data_no_shape,
        xdim=xdim,
        cutoff_lo=cutoff_lo,
        cut_borders_after_smoothing=effective_cut_borders,
        mask=mask_shape,
    )

    data_no_noise = noise_result.depth_data
    mask_noise = noise_result.mask

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------
    return FormNoiseRemovalResult(
        depth_data=data_no_noise,
        mask=mask_noise,
        relative_highest_point_location=relative_highest_point_location,
    )
