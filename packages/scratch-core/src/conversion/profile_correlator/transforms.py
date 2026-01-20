"""Transform operations for profile alignment.

This module provides functions for resampling, translation, scaling, and
cropping of 1D profiles during the alignment process. The functions handle
both single-column and multi-column profile data.

The main functions are:
- equalize_sampling_distance: Resample profiles to match pixel sizes
- make_profiles_equal_length: Symmetric cropping to equal lengths
- apply_transform: Apply translation and scaling with interpolation
- remove_boundary_zeros: Crop zero-padded borders after alignment

These correspond to the MATLAB functions:
- EqualizeSamplingDistance.m
- MakeDatasetLengthEqual.m
- TranslateScalePointset.m
- RemoveBoundaryZeros.m
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.interpolate import interp1d

from conversion.profile_correlator.data_types import Profile, TransformParameters


def equalize_sampling_distance(
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

    This corresponds to MATLAB's EqualizeSamplingDistance.m.

    :param profile_1: First profile with depth_data and pixel_size.
    :param profile_2: Second profile with depth_data and pixel_size.
    :returns: Tuple of (profile_1_out, profile_2_out) with equal pixel sizes.
        The profile that was resampled will have updated depth_data and pixel_size.

    Example::

        >>> import numpy as np
        >>> from conversion.profile_correlator.data_types import Profile
        >>> p1 = Profile(np.random.randn(100), pixel_size=0.5e-6)
        >>> p2 = Profile(np.random.randn(200), pixel_size=0.25e-6)  # Higher resolution
        >>> p1_out, p2_out = equalize_sampling_distance(p1, p2)
        >>> p1_out.pixel_size == p2_out.pixel_size  # Now equal
        True
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
        resample_factor = pixel_2 / pixel_1
        new_length = int(round(profile_2.length * resample_factor))

        # Handle single-column vs multi-column profiles
        if profile_2.depth_data.ndim == 1:
            # Single column: resample directly
            # MATLAB: resample(data, factor) for 1D
            resampled_data: NDArray[np.floating] = np.asarray(
                signal.resample(profile_2.depth_data, new_length), dtype=np.float64
            )
        else:
            # Multi-column: resample each column
            # The resampling is done along axis 0 (rows = samples)
            resampled_data = np.asarray(
                signal.resample(profile_2.depth_data, new_length, axis=0),
                dtype=np.float64,
            )

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
        resample_factor = pixel_1 / pixel_2
        new_length = int(round(profile_1.length * resample_factor))

        if profile_1.depth_data.ndim == 1:
            resampled_data_1: NDArray[np.floating] = np.asarray(
                signal.resample(profile_1.depth_data, new_length), dtype=np.float64
            )
        else:
            resampled_data_1 = np.asarray(
                signal.resample(profile_1.depth_data, new_length, axis=0),
                dtype=np.float64,
            )

        profile_1_out = Profile(
            depth_data=resampled_data_1,
            pixel_size=pixel_2,  # Now matches profile 2
            cutoff_hi=profile_1.cutoff_hi,
            cutoff_lo=profile_1.cutoff_lo,
            resolution_limit=None,
        )
        return profile_1_out, profile_2


def make_profiles_equal_length(
    profile_1: Profile | NDArray[np.floating],
    profile_2: Profile | NDArray[np.floating],
) -> tuple[Profile | NDArray[np.floating], Profile | NDArray[np.floating]]:
    """
    Crop profiles to equal length by removing samples from both ends.

    This function takes the length of the shortest profile and crops the
    longer profile to match. The cropping is symmetric: equal parts are
    removed from both ends of the longer profile.

    The function accepts either Profile objects or raw numpy arrays. If
    Profile objects are provided, the depth_data is cropped and a new
    Profile is returned. If arrays are provided, cropped arrays are returned.

    This corresponds to MATLAB's MakeDatasetLengthEqual.m.

    :param profile_1: First profile (Profile object or NDArray).
    :param profile_2: Second profile (Profile object or NDArray).
    :returns: Tuple of (profile_1_out, profile_2_out) with equal lengths.
        Returns same types as inputs (Profile or NDArray).

    Example::

        >>> import numpy as np
        >>> p1 = np.random.randn(100)
        >>> p2 = np.random.randn(120)  # 20 samples longer
        >>> p1_out, p2_out = make_profiles_equal_length(p1, p2)
        >>> len(p1_out) == len(p2_out) == 100
        True
    """
    # Extract depth data depending on input type
    is_profile_1 = isinstance(profile_1, Profile)
    is_profile_2 = isinstance(profile_2, Profile)

    data_1 = profile_1.depth_data if is_profile_1 else np.asarray(profile_1)
    data_2 = profile_2.depth_data if is_profile_2 else np.asarray(profile_2)

    # Get lengths (first dimension = number of samples)
    size_1 = data_1.shape[0]
    size_2 = data_2.shape[0]

    # If already equal, return as-is
    if size_1 == size_2:
        return profile_1, profile_2

    # Determine target length (minimum of the two)
    target_length = min(size_1, size_2)

    # Compute cropping indices for symmetric removal
    # MATLAB: 1 + floor((size - cut_index)/2) : end - ceil((size - cut_index)/2)
    # This removes equal amounts from start and end (or one more from end if odd)

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

    # Return in the same format as input
    if is_profile_1:
        profile_1_out = Profile(
            depth_data=data_1_cropped,
            pixel_size=profile_1.pixel_size,
            cutoff_hi=profile_1.cutoff_hi,
            cutoff_lo=profile_1.cutoff_lo,
            resolution_limit=profile_1.resolution_limit,
        )
    else:
        profile_1_out = data_1_cropped

    if is_profile_2:
        profile_2_out = Profile(
            depth_data=data_2_cropped,
            pixel_size=profile_2.pixel_size,
            cutoff_hi=profile_2.cutoff_hi,
            cutoff_lo=profile_2.cutoff_lo,
            resolution_limit=profile_2.resolution_limit,
        )
    else:
        profile_2_out = data_2_cropped

    return profile_1_out, profile_2_out


def apply_transform(
    profile: Profile | NDArray[np.floating],
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

    This corresponds to MATLAB's TranslateScalePointset.m.

    :param profile: Input profile (Profile object or NDArray).
    :param transforms: Single TransformParameters or sequence of them.
        Applied in order from first to last.
    :param fill_value: Value to use for positions outside the original
        profile bounds. Default is 0.0.
    :returns: Transformed profile as NDArray with same length as input.

    Example::

        >>> import numpy as np
        >>> from conversion.profile_correlator.data_types import TransformParameters
        >>> profile = np.arange(10, dtype=float)
        >>> transform = TransformParameters(translation=2.0, scaling=1.0)
        >>> result = apply_transform(profile, transform)
        >>> result[0]  # First two positions extrapolated
        0.0
    """
    # Extract depth data
    if isinstance(profile, Profile):
        data = profile.depth_data.copy()
    else:
        data = np.asarray(profile).copy()

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
    #
    # MATLAB builds: transform_matrix = [s 0 t; 0 1 0; 0 0 1] * old_matrix
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

    # Create coordinate arrays
    # MATLAB: xx = (1:length)'; xx_trans = xx * scaling + translation
    # In Python, we use 0-based indexing
    n_samples = data.shape[0]
    xx = np.arange(n_samples, dtype=np.float64)

    # Transformed coordinates: where each output position maps FROM in the input
    # If xx_trans[i] = j, then output[i] = input[j]
    xx_trans = xx * total_scaling + total_translation

    # Handle single-column vs multi-column data
    if data.ndim == 1:
        # Create interpolator for the data
        # MATLAB: interp1(xx_trans, data, xx, 'linear', 0)
        # This asks: "at positions xx, what values interpolated from (xx_trans, data)?"
        # scipy interp1d: given (x, y), return y at new x positions
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


def remove_boundary_zeros(
    profile_1: Profile | NDArray[np.floating],
    profile_2: Profile | NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """
    Remove zero-padded boundaries from two profiles.

    After alignment transformations, profiles may have zero-padding at their
    boundaries (from extrapolation beyond the original data range). This
    function finds the common non-zero region in both profiles and crops
    them to that region.

    The algorithm:
    1. Find the first and last non-zero rows in each profile
    2. Take the intersection (max of starts, min of ends)
    3. Crop both profiles to this common region

    For multi-column profiles, a row is considered zero if the sum across
    columns is zero.

    This corresponds to MATLAB's RemoveBoundaryZeros.m.

    :param profile_1: First profile (reference), Profile object or NDArray.
    :param profile_2: Second profile (compared), Profile object or NDArray.
    :returns: Tuple of (cropped_1, cropped_2, start_position) where:
        - cropped_1: First profile with boundary zeros removed
        - cropped_2: Second profile with boundary zeros removed
        - start_position: Index where the valid data begins (0-based)

    Example::

        >>> import numpy as np
        >>> p1 = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0])
        >>> p2 = np.array([0.0, 1.5, 2.5, 3.5, 0.0, 0.0])
        >>> out1, out2, start = remove_boundary_zeros(p1, p2)
        >>> start  # Common non-zero region starts at index 2
        2
        >>> len(out1) == len(out2) == 2  # Only indices 2-3 are non-zero in both
        True
    """
    # Extract depth data
    if isinstance(profile_1, Profile):
        data_1 = profile_1.depth_data
    else:
        data_1 = np.asarray(profile_1)

    if isinstance(profile_2, Profile):
        data_2 = profile_2.depth_data
    else:
        data_2 = np.asarray(profile_2)

    # For multi-column profiles, sum across columns to check for zeros
    # MATLAB: p_1 = sum(p_1, 2); which sums along dimension 2 (columns)
    if data_1.ndim > 1:
        row_sums_1 = np.sum(data_1, axis=1)
    else:
        row_sums_1 = data_1

    if data_2.ndim > 1:
        row_sums_2 = np.sum(data_2, axis=1)
    else:
        row_sums_2 = data_2

    # Create zero masks
    zero_mask_1 = row_sums_1 == 0
    zero_mask_2 = row_sums_2 == 0

    # Find start position for profile 1 (first non-zero)
    # MATLAB uses 1-based indexing and while loops
    if zero_mask_1[0]:
        # Find first non-zero
        nonzero_indices = np.where(~zero_mask_1)[0]
        if len(nonzero_indices) > 0:
            start_1 = nonzero_indices[0]
        else:
            start_1 = 0
    else:
        start_1 = 0

    # Find end position for profile 1 (last non-zero)
    if zero_mask_1[-1]:
        nonzero_indices = np.where(~zero_mask_1)[0]
        if len(nonzero_indices) > 0:
            end_1 = nonzero_indices[-1] + 1  # +1 because Python slicing is exclusive
        else:
            end_1 = len(zero_mask_1)
    else:
        end_1 = len(zero_mask_1)

    # Find start position for profile 2
    if zero_mask_2[0]:
        nonzero_indices = np.where(~zero_mask_2)[0]
        if len(nonzero_indices) > 0:
            start_2 = nonzero_indices[0]
        else:
            start_2 = 0
    else:
        start_2 = 0

    # Find end position for profile 2
    if zero_mask_2[-1]:
        nonzero_indices = np.where(~zero_mask_2)[0]
        if len(nonzero_indices) > 0:
            end_2 = nonzero_indices[-1] + 1
        else:
            end_2 = len(zero_mask_2)
    else:
        end_2 = len(zero_mask_2)

    # Find common region (intersection of valid ranges)
    # MATLAB: start_tot = max(start_1, start_2); end_tot = min(end_1, end_2)
    start_total = max(start_1, start_2)
    end_total = min(end_1, end_2)

    # Ensure valid range
    if end_total <= start_total:
        # No valid overlapping region - return empty arrays
        if data_1.ndim > 1:
            cropped_1 = data_1[0:0, :]
        else:
            cropped_1 = data_1[0:0]

        if data_2.ndim > 1:
            cropped_2 = data_2[0:0, :]
        else:
            cropped_2 = data_2[0:0]

        return cropped_1, cropped_2, start_total

    # Crop both profiles to the common region
    cropped_1 = data_1[start_total:end_total]
    cropped_2 = data_2[start_total:end_total]

    # MATLAB returns start_tot - 1 for the third output (converting to 0-based)
    # Since we're already 0-based, return start_total directly
    return cropped_1, cropped_2, start_total


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

    Example::

        >>> from conversion.profile_correlator.data_types import TransformParameters
        >>> t1 = TransformParameters(translation=5.0, scaling=1.0)
        >>> t2 = TransformParameters(translation=0.0, scaling=1.1)
        >>> trans, scale = compute_cumulative_transform([t1, t2])
        >>> trans  # 1.1 * 5.0 + 0.0 = 5.5
        5.5
        >>> scale  # 1.0 * 1.1 = 1.1
        1.1
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
