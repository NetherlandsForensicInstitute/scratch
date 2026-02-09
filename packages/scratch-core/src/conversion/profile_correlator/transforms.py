"""
Transform operations for profile alignment.

This module provides scaling functions for profile alignment:

- equalize_pixel_scale: Downsample profiles to matching pixel sizes
- apply_scaling: Apply scale transformation to a profile
"""

import numpy as np
from conversion.profile_correlator.data_types import Profile
from conversion.resample import resample_array_1d


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
        to_downsample, target_pixel_size = profile_2, profile_1.pixel_size
    else:
        to_downsample, target_pixel_size = profile_1, profile_2.pixel_size

    factor = target_pixel_size / to_downsample.pixel_size
    downsampled = Profile(
        heights=resample_array_1d(to_downsample.heights, factor),
        pixel_size=target_pixel_size,
    )

    if pixel_1 > pixel_2:
        return profile_1, downsampled
    else:
        return downsampled, profile_2
