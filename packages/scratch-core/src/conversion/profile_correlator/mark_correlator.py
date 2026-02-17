"""
Wrapper around correlate_profiles that returns aligned Mark objects.

This module provides correlate_striation_marks, a higher-level interface that
returns both comparison statistics and the aligned mark regions (2D and 1D) so
callers can visualise or further process the overlapping portion of each mark pair.
"""

import numpy as np
from scipy.signal import resample as signal_resample

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.profile_correlator.profile_correlator import (
    _calculate_idx_parameters,
    correlate_profiles,
)
from conversion.profile_correlator.transforms import equalize_pixel_scale
from conversion.profile_correlator.data_types import (
    AlignmentParameters,
    MarkCorrelationResult,
    Profile,
)
from conversion.resample import resample_array_1d


def _equalize_mark_pixel_scale(
    mark_reference: Mark, mark_compared: Mark
) -> tuple[Mark, Mark]:
    """
    Downsample the higher-resolution mark (along rows) to match pixel sizes.

    Mirrors equalize_pixel_scale for Profile, operating on the row axis (scale_x).

    :param mark_reference: Reference mark.
    :param mark_compared: Comparison mark.
    :returns: Tuple (mark_reference_out, mark_compared_out) with equal row pixel sizes.
    """
    scale_reference = mark_reference.scan_image.scale_x
    scale_compared = mark_compared.scan_image.scale_x

    if np.isclose(scale_reference, scale_compared, atol=1e-12):
        return mark_reference, mark_compared

    if scale_reference > scale_compared:
        factor = scale_reference / scale_compared
        return mark_reference, _resample_mark_rows(mark_compared, factor)
    else:
        factor = scale_compared / scale_reference
        return _resample_mark_rows(mark_reference, factor), mark_compared


def _resample_mark_rows(mark: Mark, factor: float) -> Mark:
    """
    Resample mark along the row axis (axis=0) by factor.

    factor > 1 means downsampling (fewer output rows), matching the semantics
    of resample_array_1d.

    :param mark: Input mark.
    :param factor: Scale factor for row pixel size.
    :returns: Resampled mark with updated scale_x.
    """
    data = mark.scan_image.data
    n_in = data.shape[0]
    n_out = max(1, round(n_in / factor))

    if n_out == n_in:
        return mark

    data_resampled = signal_resample(data, n_out, axis=0)
    return Mark(
        scan_image=ScanImage(
            data=np.asarray(data_resampled, dtype=np.float64),
            scale_x=mark.scan_image.scale_x * factor,
            scale_y=mark.scan_image.scale_y,
        ),
        mark_type=mark.mark_type,
        meta_data=mark.meta_data,
    )


def _trim_mark(mark: Mark, start: int, length: int) -> Mark:
    """
    Trim mark to rows [start : start + length].

    :param mark: Input mark.
    :param start: First row index.
    :param length: Number of rows to keep.
    :returns: Mark containing only the specified rows.
    """
    return Mark(
        scan_image=ScanImage(
            data=mark.scan_image.data[start : start + length, :],
            scale_x=mark.scan_image.scale_x,
            scale_y=mark.scan_image.scale_y,
        ),
        mark_type=mark.mark_type,
        meta_data=mark.meta_data,
    )


def correlate_striation_marks(
    mark_reference: Mark,
    mark_compared: Mark,
    profile_reference: Profile,
    profile_compared: Profile,
    params: AlignmentParameters = AlignmentParameters(),
) -> MarkCorrelationResult | None:
    """
    Compare two striation marks and return aligned mark regions alongside metrics.

    Designed to consume the output of preprocess_striation_mark directly:

        aligned_reference, profile_reference_mark = preprocess_striation_mark(mark_reference)
        aligned_compared, profile_compared_mark = preprocess_striation_mark(mark_compared)
        profile_reference = Profile(heights=profile_reference_mark.scan_image.data[:, 0],
                              pixel_size=profile_reference_mark.scan_image.scale_x)
        profile_compared = Profile(heights=profile_compared_mark.scan_image.data[:, 0],
                               pixel_size=profile_compared_mark.scan_image.scale_x)
        result = correlate_striation_marks(aligned_reference, aligned_compared, profile_reference, profile_compared)

    :param mark_reference: Aligned 2D scan of the reference mark.
    :param mark_compared: Aligned 2D scan of the comparison mark.
    :param profile_reference: 1D profile of the reference mark.
    :param profile_compared: 1D profile of the comparison mark.
    :param params: Alignment parameters.
    :returns: MarkCorrelationResult with aligned mark regions and metrics, or None if
        no valid alignment was found.
    """
    # 1. Correlate profiles
    results = correlate_profiles(profile_reference, profile_compared, params)
    if results is None:
        return None

    # 2. Equalize mark pixel scales (same resampling as profiles)
    mark_reference_eq, mark_compared_eq = _equalize_mark_pixel_scale(
        mark_reference, mark_compared
    )

    # 3. Re-derive alignment in sample space
    shift_samples = int(round(results.position_shift / results.pixel_size))
    scale = results.scale_factor

    # 4. Resample comp mark rows by scale
    mark_compared_scaled = _resample_mark_rows(mark_compared_eq, scale)

    # 5. Trim both marks to the overlap region
    idx_compared, idx_reference, overlap_len = _calculate_idx_parameters(
        shift_samples,
        mark_compared_scaled.scan_image.data.shape[0],
        mark_reference_eq.scan_image.data.shape[0],
    )
    mark_reference_aligned = _trim_mark(mark_reference_eq, idx_reference, overlap_len)
    mark_compared_aligned = _trim_mark(mark_compared_scaled, idx_compared, overlap_len)

    # 6. Re-derive profile overlap regions using the same indices
    profile_reference_eq, profile_compared_eq = equalize_pixel_scale(
        profile_reference, profile_compared
    )
    heights_compared_scaled = resample_array_1d(profile_compared_eq.heights, scale)
    ref_overlap = profile_reference_eq.heights[
        idx_reference : idx_reference + overlap_len
    ]
    comp_overlap = heights_compared_scaled[idx_compared : idx_compared + overlap_len]

    pixel_eq = results.pixel_size

    return MarkCorrelationResult(
        comparison_results=results,
        mark_reference_aligned=mark_reference_aligned,
        mark_compared_aligned=mark_compared_aligned,
        profile_reference_aligned=Profile(heights=ref_overlap, pixel_size=pixel_eq),
        profile_compared_aligned=Profile(heights=comp_overlap, pixel_size=pixel_eq),
    )
