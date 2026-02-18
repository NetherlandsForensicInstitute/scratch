import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.feature import match_template
from skimage.transform import warp_polar, rotate
from skimage.filters import window

from container_models.base import DepthData, FloatArray1D
from conversion.surface_comparison.models import AreaSimilarityResult, SurfaceMap

"""
Using an upsampling factor of 10 provides a registration precision of roughly 0.15–0.6μm, 
which is necessary to ensure that the initial global alignment is "close enough" for the 
subsequent per-cell search to succeed.
"""
UPSAMPLING_FACTOR = 10


def run_area_comparison(
    reference_map: SurfaceMap, comparison_map: SurfaceMap
) -> tuple[FloatArray1D, float, AreaSimilarityResult]:
    """
    Perform global automated registration of two surface maps using Fourier-Mellin transform.

    This function executes a coarse alignment to find the global rotation and translation
    required to align the comparison map with the reference map. It handles the 180-degree
    rotational ambiguity common in topographical scans (e.g., cartridge cases) by
    evaluating multiple symmetry candidates and selecting the one with the highest
    cross-correlation.

    The registration follows a two-step process:
    1. Rotation estimation via the Fourier-Mellin transform (Phase Correlation in
       log-polar space).
    2. Translation estimation via Phase Cross-Correlation (PCC) on the
       rotationally-corrected images.

    :param reference_map: The fixed surface map (SurfaceMap object).
    :param comparison_map: The moving surface map to be aligned (SurfaceMap object).
    :returns: tuple: A triplet containing:
            - translation (np.ndarray): The [dx, dy] translation vector in micrometers
              required to align the comparison map to the reference.
            - rotation (float): The optimal rotation angle in degrees.
            - similarity (AreaSimilarityResult): An object containing the global
              cross-correlation coefficient (ACCF), overlap fraction, and RMS roughness.

    :raises: ValueError: If either map lacks an 'unfiltered_height_map'.

    Notes:
        - The translation is calculated in the reference coordinate system.
        - Rotations are performed around the image center.
        - The 'overlap_fraction' accounts for data lost due to rotation and shift
          using NaN-padding (cval=np.nan).
    """
    _require_unfiltered_map(reference_map, "reference")
    _require_unfiltered_map(comparison_map, "comparison")

    base_theta = _estimate_rotation(reference_map, comparison_map)
    candidates = {
        0.0,
        base_theta,
        -base_theta,
        (base_theta + 180) % 360,
        (base_theta - 180) % 360,
    }

    # Initialize baseline with 0.0 rotation
    best_rot = 0.0
    best_trans = _estimate_translation(
        reference_map.height_map,
        rotate(comparison_map.height_map, 0.0, preserve_range=True, cval=np.nan),  # type: ignore
        reference_map.pixel_spacing,
    )
    best_sim = _evaluate_similarity_internal(
        reference_map, comparison_map, 0.0, best_trans
    )
    for angle in candidates:
        if np.isclose(angle, 0.0):
            continue

        # Rotate with NaN padding to ensure overlap fraction is calculated correctly
        comp_rotated = rotate(
            comparison_map.height_map,
            -angle,
            preserve_range=True,
            cval=np.nan,  # type: ignore
        )
        trans = _estimate_translation(
            reference_map.height_map, comp_rotated, reference_map.pixel_spacing
        )
        sim = _evaluate_similarity_internal(reference_map, comparison_map, angle, trans)
        if sim.cross_correlation_coefficient > best_sim.cross_correlation_coefficient:
            best_sim = sim
            best_rot = angle
            best_trans = trans

    return best_trans, float(best_rot), best_sim


def _estimate_rotation(ref_map: SurfaceMap, comp_map: SurfaceMap) -> float:
    """
    Estimate the relative rotation between two maps using the Fourier-Mellin transform.

    The method converts the power spectra of both maps into log-polar coordinates.
    In this domain, a rotation in the spatial domain becomes a translation in the
    polar domain, which is then solved using Phase Cross-Correlation.

    :param ref_map: The reference surface map.
    :param comp_map: The comparison surface map.
    :returns: The estimated rotation angle in degrees required to align the
              comparison map with the reference orientation.
    """

    def get_log_spectrum(data):
        clean = np.nan_to_num(data, nan=np.nanmean(data))
        clean -= np.mean(clean)
        w_data = clean * window("hann", clean.shape)
        spec = np.abs(np.fft.fftshift(np.fft.fft2(w_data)))
        return np.log1p(spec)

    spec_ref = get_log_spectrum(ref_map.height_map)
    spec_comp = get_log_spectrum(comp_map.height_map)

    radius = min(spec_ref.shape) // 2
    warped_ref = warp_polar(spec_ref, radius=radius, output_shape=(360, 200))
    warped_comp = warp_polar(spec_comp, radius=radius, output_shape=(360, 200))

    shifts, _, _ = phase_cross_correlation(
        warped_ref, warped_comp, upsample_factor=UPSAMPLING_FACTOR
    )
    return float(shifts[0])


def _estimate_translation(
    ref_data: DepthData, comp_data: DepthData, spacing: FloatArray1D
) -> np.ndarray:
    """
    Find the relative [dx, dy] translation between two height maps using Phase Cross-Correlation.

    The translation is estimated by finding the peak of the cross-correlation
    surface between the two datasets. The result represents the spatial shift
    needed to bring the comparison data into alignment with the reference data.

    :param ref_data: 2D array of height values for the reference surface,
        shape (rows, columns).
    :param comp_data: 2D array of height values for the comparison surface,
        shape (rows, columns).
    :param spacing: The pixel spacing [dx, dy] in micrometers, shape (2,).
    :returns: The translation vector [dx, dy] in micrometers, shape (2,).
    """
    r = np.nan_to_num(ref_data, nan=np.nanmean(ref_data))  # type: ignore
    c = np.nan_to_num(comp_data, nan=np.nanmean(comp_data))  # type: ignore
    shift, _, _ = phase_cross_correlation(r, c, upsample_factor=UPSAMPLING_FACTOR)
    return np.array([shift[1], shift[0]]) * spacing


def _evaluate_similarity_internal(
    ref_map: SurfaceMap, comp_map: SurfaceMap, rot: float, trans: FloatArray1D
) -> AreaSimilarityResult:
    """
    Evaluate similarity metrics between two surface maps at a specific alignment.

    This function applies the specified rotation and translation to the comparison
    map, then calculates the global Area Cross-Correlation Function (ACCF) score
    and the physical overlap fraction between valid (non-NaN) data regions.

    :param ref_map: The reference SurfaceMap object containing the height data.
    :param comp_map: The comparison SurfaceMap object to be evaluated.
    :param rot: The rotation angle in degrees to apply to the comparison map.
    :param trans: The translation vector [dx, dy] in micrometers, shape (2,).
    :returns: An AreaSimilarityResult containing the cross-correlation
              coefficient, overlap fraction, and RMS roughness (Sq) values.
    """

    def _sq(data):
        v = data[~np.isnan(data)]
        return float(np.sqrt(np.mean((v - np.mean(v)) ** 2))) if v.size > 0 else 0.0

    ref_raw = ref_map.unfiltered_height_map

    # 1. Rotate using NaN for empty space
    comp_raw = rotate(
        comp_map.unfiltered_height_map,
        -rot,
        preserve_range=True,
        cval=np.nan,  # type: ignore
    )

    # 2. Compute Overlap Fraction
    mask_ref = ~np.isnan(ref_raw)  # type: ignore
    mask_comp = ~np.isnan(comp_raw)  # type: ignore
    pixel_shift = (trans / ref_map.pixel_spacing).astype(int)
    mask_comp_shifted = np.roll(
        mask_comp, shift=(pixel_shift[1], pixel_shift[0]), axis=(0, 1)
    )
    overlap_area = np.logical_and(mask_ref, mask_comp_shifted)
    overlap_fraction = np.count_nonzero(overlap_area) / np.count_nonzero(mask_ref)

    # 3. Correlation
    ref_clean = np.nan_to_num(ref_raw, nan=np.nanmean(ref_raw))  # type: ignore
    comp_clean = np.nan_to_num(comp_raw, nan=np.nanmean(comp_raw))  # type: ignore
    accf_map = match_template(comp_clean, ref_clean, pad_input=True)
    global_cc = float(np.max(accf_map))

    return AreaSimilarityResult(
        cross_correlation_coefficient=global_cc,
        overlap_fraction=float(overlap_fraction),
        reference_root_mean_square_roughness=_sq(ref_raw),
        comparison_root_mean_square_roughness=_sq(comp_raw),
    )


def _require_unfiltered_map(surface_map: SurfaceMap, label: str) -> None:
    if surface_map.unfiltered_height_map is None:
        raise ValueError(f"The {label} surface map has no unfiltered height data.")
