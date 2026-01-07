"""Correct striation marks for bullet surface curvature (unfolding).

This module provides functionality to correct for the curved surface of a
bullet shell when analyzing striation marks. The bullet surface is curved
along its longitudinal axis, and this curvature distorts the striation
pattern. Unfolding compensates for this by interpolating the data onto
a flat reference grid.

The algorithm works in four steps:
    1. Bandpass filter to extract striation marks (remove shape and noise)
    2. Lowpass filter to get the global bullet shape/curvature
    3. Calculate curvature angles and unfold the surface
    4. Interpolate data onto the unfolded grid

Migrated from MATLAB: UnfoldBullet.m
"""

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from conversion.cheby_cutoff_to_gauss_sigma import cheby_cutoff_to_gauss_sigma
from conversion.remove_noise_gaussian import remove_noise_gaussian
from conversion.remove_shape_gaussian import remove_shape_gaussian


@dataclass
class UnfoldBulletResult:
    """Result container for bullet unfolding operation.

    Attributes:
        depth_data: The unfolded depth data. This is placeholder for now as
            the original MATLAB didn't update the main output properly.
        striations: The extracted striation marks (bandpass filtered and cropped).
            This is the primary output used for comparison.
        mask: Boolean mask for valid data regions (may be None if no masking).
        relative_highest_point_location: Relative position (0 to 1) of the
            highest point on the bullet surface. This indicates how well
            the bullet was positioned in the scanner. A value near 0.5
            means the highest point is in the middle of the scan.
    """

    depth_data: NDArray[np.floating]
    striations: NDArray[np.floating]
    mask: NDArray[np.bool_] | None
    relative_highest_point_location: float


def _compute_gradient(
    data: NDArray[np.floating], axis: int = 0
) -> NDArray[np.floating]:
    """
    Compute gradient (derivative) of data along specified axis.

    Uses central differences for interior points and forward/backward
    differences at boundaries.

    :param data: Input array.
    :param axis: Axis along which to compute gradient.
    :return: Gradient array with same shape as input.
    """
    return np.gradient(data, axis=axis)


def unfold_bullet(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_hi: float,
    cutoff_lo: float = 1000.0,
    cut_borders_after_smoothing: bool = False,
    mask: NDArray[np.bool_] | None = None,
) -> UnfoldBulletResult:
    """
    Correct striation marks for bullet surface curvature.

    This function compensates for the curved surface of a bullet shell when
    analyzing striation marks. The bullet surface is curved, which causes
    the spacing between measurements to vary depending on the local slope.
    Unfolding corrects for this by interpolating the data onto a flat
    reference grid with uniform spacing.

    The algorithm has four main steps:

    **Step 1: Extract striation marks**
        Apply bandpass filtering (shape removal then noise removal) to
        isolate the striation pattern from both large-scale curvature
        and fine noise.

    **Step 2: Get global bullet shape**
        Apply lowpass filtering to the original data to extract the
        smooth bullet curvature profile without striations.

    **Step 3: Calculate curvature and unfold**
        - Compute the gradient (slope) of the smoothed surface
        - Convert slopes to angles in degrees
        - Calculate the local grid spacing correction: 1/cos(angle)
        - Find the highest point (location of minimum gradient)

    **Step 4: Interpolate onto unfolded grid**
        - Build cumulative distance grid from spacing corrections
        - Shift grid so highest point stays fixed
        - Interpolate each column onto the new uniform grid

    :param depth_data: 2D depth/height data array. Rows should be along
        the bullet's longitudinal axis (direction of curvature).
    :param xdim: Pixel spacing in meters (m). Distance between adjacent
        measurements in the scan.
    :param cutoff_hi: High-frequency cutoff wavelength in micrometers (um)
        for shape removal. Typically 2000 um.
    :param cutoff_lo: Low-frequency cutoff wavelength in micrometers (um)
        for extracting global shape. Default 1000 um preserves the
        overall bullet curvature while removing fine detail.
    :param cut_borders_after_smoothing: If True, crop borders after
        filtering. Default False for unfolding to preserve data extent.
    :param mask: Optional boolean mask array (True = valid data).
        Currently not fully implemented - passed through.
    :return: UnfoldBulletResult containing:
        - depth_data: Unfolded striation data
        - striations: Cropped striation marks (before unfolding)
        - mask: Output mask (may be None)
        - relative_highest_point_location: Position of highest point (0-1)

    Example:
        >>> import numpy as np
        >>> # Create synthetic curved bullet surface with striations
        >>> rows = np.linspace(-10, 10, 500)
        >>> curvature = -0.1 * rows**2  # Parabolic curve (convex)
        >>> striations = np.sin(2 * np.pi * rows / 5) * 0.01  # Small striations
        >>> surface = np.tile((curvature + striations).reshape(-1, 1), (1, 100))
        >>> # Unfold the bullet
        >>> result = unfold_bullet(surface, xdim=1e-6, cutoff_hi=2000)
        >>> print(f"Highest point at: {result.relative_highest_point_location:.2%}")
        Highest point at: 50.00%

    Note:
        - The highest point detection uses a margin of 150 pixels from edges
        - Unfolding is performed column-by-column using linear interpolation
        - The highest point position is fixed during unfolding to prevent drift
        - The function matches MATLAB UnfoldBullet.m behavior
    """
    # Ensure 2D input
    depth_data = np.atleast_2d(depth_data)
    if depth_data.shape[0] == 1:
        depth_data = depth_data.T

    # Initialize output mask
    mask_out: NDArray[np.bool_] | None = None

    # -------------------------------------------------------------------------
    # STEP 1: Get striation marks (bandpass filtering)
    # -------------------------------------------------------------------------
    # First remove shape (highpass), then remove noise (lowpass) = bandpass
    # This isolates the striation features from both curvature and fine noise

    # Shape removal: removes large-scale curvature
    shape_result = remove_shape_gaussian(
        depth_data,
        xdim=xdim,
        cutoff_hi=cutoff_hi,
        cut_borders_after_smoothing=False,  # Don't crop yet
        mask=mask,
    )

    # Noise removal: removes high-frequency noise from the residuals
    noise_result = remove_noise_gaussian(
        shape_result.depth_data,
        xdim=xdim,
        cutoff_lo=cutoff_lo if cutoff_lo > 0 else 250.0,  # Default noise cutoff
        cut_borders_after_smoothing=False,
        mask=shape_result.mask,
    )

    # The bandpass result contains just the striation marks
    data_striations = noise_result.depth_data

    # Calculate sigma for cropping
    sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)
    sigma_int = int(ceil(sigma))

    # Crop the striation data
    if sigma_int > 0 and data_striations.shape[0] > 2 * sigma_int:
        striations_cropped = data_striations[sigma_int:-sigma_int, :]
    else:
        striations_cropped = data_striations

    # -------------------------------------------------------------------------
    # STEP 2: Get global shape and determine highest point
    # -------------------------------------------------------------------------
    # Apply lowpass filter to get smooth bullet curvature (remove striations)

    smooth_result = remove_noise_gaussian(
        depth_data,
        xdim=xdim,
        cutoff_lo=1000.0,  # Fixed cutoff for shape extraction
        cut_borders_after_smoothing=False,
        mask=mask,
    )

    # Crop the smoothed data to match striation cropping
    if sigma_int > 0 and smooth_result.depth_data.shape[0] > 2 * sigma_int:
        data_smoothed = smooth_result.depth_data[sigma_int:-sigma_int, :]
    else:
        data_smoothed = smooth_result.depth_data

    # Calculate the angle of curvature at each position
    # Gradient gives the slope (dz/dx), divide by xdim to get proper units
    gradient = _compute_gradient(data_smoothed, axis=0) / xdim

    # Convert slope to angle in degrees
    gradient_angles = np.degrees(np.arctan(gradient))

    # Calculate new grid spacing based on curvature
    # When surface is tilted, distance between points on surface is 1/cos(angle)
    new_grid_spacing = 1.0 / np.cos(np.radians(gradient_angles))

    # Determine highest point (location with lowest gradient magnitude)
    # Use margin of 150 pixels from edges to avoid boundary effects
    margin = 150
    n_rows = gradient_angles.shape[0]

    if n_rows > 2 * margin:
        # Sum absolute gradient across columns for each row
        gradient_sum = np.sum(
            np.abs(gradient_angles[margin : n_rows - margin, :]), axis=1
        )
        # Find row with minimum total gradient (flattest = highest point)
        min_idx = np.argmin(gradient_sum)
        highest_point_idx = min_idx + margin
    else:
        # Data too small for margins, find minimum in full range
        gradient_sum = np.sum(np.abs(gradient_angles), axis=1)
        highest_point_idx = np.argmin(gradient_sum)

    # Calculate relative position (0 to 1)
    relative_highest_point_location = float(highest_point_idx) / float(n_rows)

    # -------------------------------------------------------------------------
    # STEP 3: Unfold the bullet
    # -------------------------------------------------------------------------
    # Build cumulative distance grid from spacing corrections
    xnew = np.cumsum(new_grid_spacing, axis=0)

    # Calculate new data size (based on maximum unfolded length)
    new_size = int(ceil(np.max(xnew[-1, :])))

    # Allocate output array
    new_data = np.zeros((new_size, striations_cropped.shape[1]))

    # Create uniform output grid
    x_profile = np.arange(1, new_size + 1).astype(float)

    # Shift xnew so that the highest point stays at its original position
    # This ensures unfolding happens outward from the highest point
    xnew_shifted = xnew - xnew[highest_point_idx, :] + highest_point_idx

    # Interpolate each column onto the new grid
    for col in range(striations_cropped.shape[1]):
        new_data[:, col] = np.interp(
            x_profile,
            xnew_shifted[:, col],
            striations_cropped[:, col],
            left=0,
            right=0,
        )

    # -------------------------------------------------------------------------
    # STEP 4: Prepare output
    # -------------------------------------------------------------------------
    return UnfoldBulletResult(
        depth_data=new_data,
        striations=striations_cropped,
        mask=mask_out,
        relative_highest_point_location=relative_highest_point_location,
    )
