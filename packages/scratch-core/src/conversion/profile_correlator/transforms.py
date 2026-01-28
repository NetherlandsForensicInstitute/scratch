"""
Transform operations for profile alignment.

This module provides functions for resampling, translation, scaling, and
cropping of 1D profiles during the alignment process. The functions handle
both single-column and multi-column profile data.

The main functions are:
- equalize_sampling_distance: Resample profiles to match pixel sizes
- make_profiles_equal_length: Symmetric cropping to equal lengths
- apply_transform: Apply translation and scaling with interpolation

All functions operate on Profile objects. For raw array operations, extract
the depth_data from Profile objects.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from conversion.profile_correlator.data_types import Profile, TransformParameters


def _cubic_bspline(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate the cubic B-spline basis function.

    This is the 4-point-support kernel used by DIPimage's ``resample``
    for order-3 interpolation (direct evaluation, no prefilter).

    :param x: Array of evaluation positions.
    :returns: Kernel values (0 for |x| >= 2).
    """
    ax = np.abs(x)
    result = np.zeros_like(ax)
    m1 = ax < 1.0
    result[m1] = (3.0 * ax[m1] ** 3 - 6.0 * ax[m1] ** 2 + 4.0) / 6.0
    m2 = (ax >= 1.0) & (ax < 2.0)
    result[m2] = (2.0 - ax[m2]) ** 3 / 6.0
    return result


def _resample_1d(
    data: NDArray[np.floating],
    zoom: float,
) -> NDArray[np.floating]:
    """
    Resample a 1D array using cubic B-spline interpolation (DIPimage-compatible).

    Output sample *j* maps to input position ``j / zoom``.

    :param data: 1D input array.
    :param zoom: Zoom factor (< 1 for downsampling, > 1 for upsampling).
    :returns: Resampled 1D array of length ``max(1, round(len(data) * zoom))``.
    """
    n_in = len(data)
    n_out = max(1, int(round(n_in * zoom)))

    if n_out == n_in:
        return data.copy()

    new_x = np.arange(n_out, dtype=np.float64) / zoom
    result = np.empty(n_out, dtype=np.float64)

    for j in range(n_out):
        x = new_x[j]
        i0 = int(np.floor(x))
        val = 0.0
        for di in range(-1, 3):
            idx = i0 + di
            idx_clamped = max(0, min(n_in - 1, idx))
            weight = float(_cubic_bspline(np.array([x - idx]))[0])
            val += data[idx_clamped] * weight
        result[j] = val

    return result


def equalize_pixel_scale(
    profile_1: Profile,
    profile_2: Profile,
) -> tuple[Profile, Profile]:
    """
    Resample profiles to have equal pixel sizes.

    This function compares the pixel sizes (sampling distances) of two profiles
    and resamples the one with smaller pixel size (higher resolution) to match
    the one with larger pixel size (lower resolution). The resampling uses
    scipy.signal.resample which applies Fourier-based interpolation.

    The profile with the larger pixel size is returned unchanged, while the
    other profile is resampled. After resampling, the resolution_limit field
    is cleared as it may no longer be valid.

    :param profile_1: First profile with depth_data and pixel_size.
    :param profile_2: Second profile with depth_data and pixel_size.
    :returns: Tuple of (profile_1_out, profile_2_out) with equal pixel sizes.
        The profile that was resampled will have updated depth_data and pixel_size.
    """
    # Get pixel sizes
    pixel_1 = profile_1.pixel_size
    pixel_2 = profile_2.pixel_size

    # If pixel sizes are already equal, return copies
    if pixel_1 == pixel_2:
        return profile_1, profile_2

    # Determine which profile needs resampling (the one with smaller pixel size)
    # Resample to the larger pixel size (lower resolution)
    if pixel_1 > pixel_2:
        # Profile 2 has higher resolution, resample it to match profile 1
        zoom = pixel_2 / pixel_1

        # Handle single-column vs multi-column profiles
        if profile_2.depth_data.ndim == 1:
            resampled_data: NDArray[np.floating] = _resample_1d(
                profile_2.depth_data, zoom
            )
        else:
            # Multi-column: resample each column
            cols = [
                _resample_1d(profile_2.depth_data[:, c], zoom)
                for c in range(profile_2.depth_data.shape[1])
            ]
            resampled_data = np.column_stack(cols)

        # Create new profile with updated data and pixel size
        # Note: resolution_limit is cleared as it may no longer be valid
        profile_2_out = Profile(
            depth_data=resampled_data,
            pixel_size=pixel_1,  # Now matches profile 1
            cutoff_hi=profile_2.cutoff_hi,
            cutoff_lo=profile_2.cutoff_lo,
            resolution_limit=None,  # Clear LR after resampling
        )
        return profile_1, profile_2_out

    else:
        # Profile 1 has higher resolution, resample it to match profile 2
        zoom = pixel_1 / pixel_2

        if profile_1.depth_data.ndim == 1:
            resampled_data_1: NDArray[np.floating] = _resample_1d(
                profile_1.depth_data, zoom
            )
        else:
            cols = [
                _resample_1d(profile_1.depth_data[:, c], zoom)
                for c in range(profile_1.depth_data.shape[1])
            ]
            resampled_data_1 = np.column_stack(cols)

        profile_1_out = Profile(
            depth_data=resampled_data_1,
            pixel_size=pixel_2,  # Now matches profile 2
            cutoff_hi=profile_1.cutoff_hi,
            cutoff_lo=profile_1.cutoff_lo,
            resolution_limit=None,
        )
        return profile_1_out, profile_2


def make_profiles_equal_length(
    profile_1: Profile,
    profile_2: Profile,
) -> tuple[Profile, Profile]:
    """
    Crop profiles to equal length by removing samples from both ends.

    This function takes the length of the shortest profile and crops the
    longer profile to match. The cropping is symmetric: equal parts are
    removed from both ends of the longer profile.

    :param profile_1: First profile.
    :param profile_2: Second profile.
    :returns: Tuple of (profile_1_out, profile_2_out) with equal lengths.
    """
    # Get depth data
    data_1 = profile_1.depth_data
    data_2 = profile_2.depth_data

    # Get lengths (first dimension = number of samples)
    size_1 = data_1.shape[0]
    size_2 = data_2.shape[0]

    # If already equal, return as-is
    if size_1 == size_2:
        return profile_1, profile_2

    # Determine target length (minimum of the two)
    target_length = min(size_1, size_2)

    # Compute cropping indices for symmetric removal

    # Profile 1 cropping
    diff_1 = size_1 - target_length
    start_1 = diff_1 // 2  # floor
    end_1 = size_1 - (diff_1 - start_1)  # ceil for the end removal
    data_1_cropped = data_1[start_1:end_1]

    # Profile 2 cropping
    diff_2 = size_2 - target_length
    start_2 = diff_2 // 2
    end_2 = size_2 - (diff_2 - start_2)
    data_2_cropped = data_2[start_2:end_2]

    # Create new Profile objects with cropped data
    profile_1_out = Profile(
        depth_data=data_1_cropped,
        pixel_size=profile_1.pixel_size,
        cutoff_hi=profile_1.cutoff_hi,
        cutoff_lo=profile_1.cutoff_lo,
        resolution_limit=profile_1.resolution_limit,
    )

    profile_2_out = Profile(
        depth_data=data_2_cropped,
        pixel_size=profile_2.pixel_size,
        cutoff_hi=profile_2.cutoff_hi,
        cutoff_lo=profile_2.cutoff_lo,
        resolution_limit=profile_2.resolution_limit,
    )

    return profile_1_out, profile_2_out


def apply_transform(
    profile: Profile,
    transforms: Sequence[TransformParameters] | TransformParameters,
    fill_value: float = 0.0,
) -> NDArray[np.floating]:
    """
    Apply translation and scaling transformation to a profile.

    This function transforms a profile using a sequence of translation and
    scaling parameters. Multiple transforms are composed by multiplying their
    transformation matrices. The transformed profile is computed using linear
    interpolation, with extrapolated values filled with fill_value.

    The transformation model is:
        x' = scaling * x + translation

    where x is the original sample index and x' is the transformed index.
    The interpolation finds the value at the original x corresponding to
    each new position.

    For multiple transforms, they are composed in order:
        T_total = T_n * T_{n-1} * ... * T_1

    :param profile: Input profile.
    :param transforms: Single TransformParameters or sequence of them.
        Applied in order from first to last.
    :param fill_value: Value to use for positions outside the original
        profile bounds. Default is 0.0.
    :returns: Transformed profile as NDArray with same length as input.
    """
    # Extract depth data
    data = profile.depth_data.copy()

    # Ensure transforms is a sequence
    if isinstance(transforms, TransformParameters):
        transforms = [transforms]

    # Handle empty transforms
    if len(transforms) == 0:
        return data

    # Compose transformation matrices
    # Each transform is: [scaling, 0, translation; 0, 1, 0; 0, 0, 1]
    # Composition: T_new * T_old gives combined transform
    #
    # For transforms [t1, t2, ...]:
    #   x' = s2 * (s1 * x + t1) + t2 = s2*s1*x + s2*t1 + t2
    total_scaling = 1.0
    total_translation = 0.0

    for i, t in enumerate(transforms):
        if i == 0:
            total_scaling = t.scaling
            total_translation = t.translation
        else:
            # Compose: new transform applied after previous
            # x'' = s_new * x' + t_new = s_new * (s_old * x + t_old) + t_new
            #     = (s_new * s_old) * x + (s_new * t_old + t_new)
            total_translation = t.scaling * total_translation + t.translation
            total_scaling = t.scaling * total_scaling

    # Create coordinate arrays using 1-based indexing to match MATLAB's
    # TranslateScalePointset: xx = (1:length(pointset))'
    n_samples = data.shape[0]
    xx = np.arange(1, n_samples + 1, dtype=np.float64)

    # Transformed coordinates: where each output position maps FROM in the input
    # MATLAB: xx_trans = xx * transform_matrix(1,1) + transform_matrix(1,3)
    xx_trans = xx * total_scaling + total_translation

    # Handle single-column vs multi-column data
    if data.ndim == 1:
        # Create interpolator for the data
        interpolator = interp1d(
            xx_trans,
            data,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )
        transformed = interpolator(xx)
    else:
        # Multi-column: interpolate each column separately
        transformed = np.zeros_like(data)
        for col in range(data.shape[1]):
            interpolator = interp1d(
                xx_trans,
                data[:, col],
                kind="linear",
                bounds_error=False,
                fill_value=fill_value,
            )
            transformed[:, col] = interpolator(xx)

    return transformed


def compute_cumulative_transform(
    transforms: Sequence[TransformParameters],
) -> tuple[float, float]:
    """
    Compute the cumulative translation and scaling from a sequence of transforms.

    This function composes multiple transformation parameters into a single
    equivalent transformation. The transforms are applied in order, with
    each subsequent transform acting on the result of the previous.

    The composition follows:
        x'' = s2 * (s1 * x + t1) + t2 = (s2 * s1) * x + (s2 * t1 + t2)

    :param transforms: Sequence of TransformParameters to compose.
    :returns: Tuple of (total_translation, total_scaling).
    """
    if len(transforms) == 0:
        return 0.0, 1.0

    total_translation = 0.0
    total_scaling = 1.0

    for i, t in enumerate(transforms):
        if i == 0:
            total_translation = t.translation
            total_scaling = t.scaling
        else:
            # Compose transforms
            total_translation = t.scaling * total_translation + t.translation
            total_scaling = t.scaling * total_scaling

    return total_translation, total_scaling
