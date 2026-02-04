"""
Transform operations for profile alignment.

This module provides scaling functions for profile alignment:

- equalize_pixel_scale: Downsample profiles to matching pixel sizes
- apply_scaling: Apply scale transformation to a profile
"""

import numpy as np
from scipy.interpolate import interp1d

from container_models.base import FloatArray1D
from conversion.profile_correlator.data_types import Profile
from conversion.resample import resample_array_1d


def apply_scaling(
    data: FloatArray1D,
    scale_factor: float,
) -> FloatArray1D:
    """
    Resample a profile at scaled positions to simulate stretching/compression.

    Samples at positions [0, scale, 2*scale, ...] using cubic interpolation.
    Output has the same length as input. Positions outside the input range
    are filled with zeros.

    Used during profile alignment to test how well profiles match at different
    scale factors (e.g., due to measurement differences between firearms).

    :param data: Input 1D profile data.
    :param scale_factor: Sampling interval. >1 compresses the pattern (more of
        source consumed), <1 stretches the pattern (less of source consumed).
    :returns: Scaled profile array (same length as input). Values at positions
        beyond the original data range are filled with zeros.
    """
    if np.isclose(scale_factor, 1.0, atol=1e-12):
        return data.copy()

    n = len(data)
    x_orig = np.arange(n, dtype=np.float64)
    new_positions = x_orig * scale_factor

    interpolator = interp1d(
        x_orig,
        data,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
    )

    return interpolator(new_positions)


def equalize_pixel_scale(
    profile_1: Profile,
    profile_2: Profile,
) -> tuple[Profile, Profile]:
    """
    Downsample the higher-resolution profile to match pixel sizes.

    This function compares the pixel sizes (sampling distances) of two profiles
    and downsamples the one with smaller pixel size (higher resolution) to match
    the one with larger pixel size (lower resolution). The downsampling uses
    cubic spline interpolation.

    The profile with the larger pixel size is returned unchanged, while the
    other profile is downsampled.

    :param profile_1: First profile with heights and pixel_size.
    :param profile_2: Second profile with heights and pixel_size.
    :returns: Tuple of (profile_1_out, profile_2_out) with equal pixel sizes.
        The profile that was downsampled will have updated heights and pixel_size.
    """
    pixel_1 = profile_1.pixel_size
    pixel_2 = profile_2.pixel_size

    if np.isclose(pixel_1, pixel_2, atol=1e-12):
        return profile_1, profile_2

    # Downsample the higher-resolution profile to match the lower-resolution one
    if pixel_1 > pixel_2:
        to_downsample, other = profile_2, profile_1
    else:
        to_downsample, other = profile_1, profile_2

    target_pixel_size = other.pixel_size
    factor = to_downsample.pixel_size / target_pixel_size
    downsampled = Profile(
        heights=resample_array_1d(to_downsample.heights, factor),
        pixel_size=target_pixel_size,
    )

    if pixel_1 > pixel_2:
        return profile_1, downsampled
    else:
        return downsampled, profile_2
