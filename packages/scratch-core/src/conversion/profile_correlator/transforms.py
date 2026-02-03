"""
Transform operations for profile alignment.

This module provides resampling and scaling functions for profile alignment:

- equalize_pixel_scale: Resample profiles to matching pixel sizes
- apply_scaling: Apply scale transformation to a profile
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from conversion.profile_correlator.data_types import Profile


def _interpolate_1d(
    data: NDArray[np.floating],
    new_positions: NDArray[np.floating],
    fill_value: float = 0.0,
) -> NDArray[np.floating]:
    """
    Interpolate 1D data at new positions using cubic spline interpolation.

    :param data: Input 1D array with values at integer positions [0, 1, ..., n-1].
    :param new_positions: Positions at which to sample (can be fractional).
    :param fill_value: Value to use for positions outside the data range.
    :returns: Interpolated values at the new positions.
    """
    n = len(data)
    x_orig = np.arange(n, dtype=np.float64)

    interpolator = interp1d(
        x_orig,
        data,
        kind="cubic",
        bounds_error=False,
        fill_value=fill_value,
    )

    return interpolator(new_positions)


def _resample_1d(
    data: NDArray[np.floating],
    zoom: float,
) -> NDArray[np.floating]:
    """
    Resample a 1D array to a new length using cubic spline interpolation.

    Output sample j maps to input position j / zoom.

    :param data: 1D input array.
    :param zoom: Zoom factor (< 1 for downsampling, > 1 for upsampling).
    :returns: Resampled 1D array of length max(1, round(len(data) * zoom)).
    """
    n_in = len(data)
    n_out = max(1, int(round(n_in * zoom)))

    if n_out == n_in:
        return data.copy()

    # Map output positions to input positions
    new_positions = np.arange(n_out, dtype=np.float64) / zoom

    return _interpolate_1d(data, new_positions)


def apply_scaling(
    data: NDArray[np.floating],
    scale_factor: float,
) -> NDArray[np.floating]:
    """
    Apply scaling transformation to a profile.

    Scale factor > 1.0 stretches the profile (samples later positions in data).
    Scale factor < 1.0 compresses the profile (samples earlier positions in data).

    :param data: Input 1D profile data.
    :param scale_factor: Scaling factor (1.0 = no scaling).
    :returns: Scaled profile with same length as input.
    """
    if scale_factor == 1.0:
        return data.copy()

    n = len(data)

    # Sample at scaled positions (1-indexed for MATLAB compatibility)
    # scale > 1.0: sample later (stretch)
    # scale < 1.0: sample earlier (compress)
    x_orig = np.arange(n, dtype=np.float64)
    new_positions = x_orig * scale_factor

    return _interpolate_1d(data, new_positions)


def equalize_pixel_scale(
    profile_1: Profile,
    profile_2: Profile,
) -> tuple[Profile, Profile]:
    """
    Resample profiles to have equal pixel sizes.

    This function compares the pixel sizes (sampling distances) of two profiles
    and resamples the one with smaller pixel size (higher resolution) to match
    the one with larger pixel size (lower resolution). The resampling uses
    cubic spline interpolation.

    The profile with the larger pixel size is returned unchanged, while the
    other profile is resampled.

    :param profile_1: First profile with depth_data and pixel_size.
    :param profile_2: Second profile with depth_data and pixel_size.
    :returns: Tuple of (profile_1_out, profile_2_out) with equal pixel sizes.
        The profile that was resampled will have updated depth_data and pixel_size.
    """
    # Get pixel sizes
    pixel_1 = profile_1.pixel_size
    pixel_2 = profile_2.pixel_size

    # If pixel sizes are already equal, return unchanged
    if pixel_1 == pixel_2:
        return profile_1, profile_2

    # Determine which profile needs resampling (the one with smaller pixel size)
    # Resample to the larger pixel size (lower resolution)
    if pixel_1 > pixel_2:
        # Profile 2 has higher resolution, resample it to match profile 1
        zoom = pixel_2 / pixel_1
        resampled_data = _resample_1d(profile_2.depth_data, zoom)

        profile_2_out = Profile(
            depth_data=resampled_data,
            pixel_size=pixel_1,  # Now matches profile 1
        )
        return profile_1, profile_2_out

    else:
        # Profile 1 has higher resolution, resample it to match profile 2
        zoom = pixel_1 / pixel_2
        resampled_data = _resample_1d(profile_1.depth_data, zoom)

        profile_1_out = Profile(
            depth_data=resampled_data,
            pixel_size=pixel_2,  # Now matches profile 2
        )
        return profile_1_out, profile_2
