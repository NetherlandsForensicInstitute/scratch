"""
CMC classification using the median-based algorithm (MATLAB procedure 6).

This replaces the pairwise consensus voting with the correct median + outlier
rejection algorithm that matches the MATLAB ``cell_cmc_median`` procedure 6,
which is the default for ``MAX_SIMILARITY_MEDIAN``.
"""

import numpy as np
from conversion.surface_comparison.models import ComparisonResult, ComparisonParams
from conversion.surface_comparison.alignment_utils import detect_outliers_esd


def _circular_median(angles: np.ndarray) -> float:
    """
    Compute the circular median of a set of angles (in radians).

    The circular median minimises the sum of absolute angular distances.
    For small angular spreads (typical in CMC analysis) this is equivalent
    to unwrapping and taking the ordinary median, but we use the robust
    circular definition to match MATLAB ``angle_median``.

    :param angles: 1-D array of angles in radians.
    :returns: The circular median angle in radians.
    """
    angles = angles[~np.isnan(angles)]
    if angles.size == 0:
        return np.nan

    # For each candidate (every observed angle), compute the sum of
    # absolute circular distances.  The candidate that minimises this
    # is the circular median.
    best_idx = 0
    best_cost = np.inf
    for i, candidate in enumerate(angles):
        cost = np.sum(
            np.abs(
                np.arctan2(
                    np.sin(angles - candidate),
                    np.cos(angles - candidate),
                )
            )
        )
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return float(angles[best_idx])


def _wrapped_angle_diff(angles: np.ndarray, reference: float) -> np.ndarray:
    """
    Signed angular difference wrapped to [-pi, pi].

    Matches MATLAB ``angle_diff(angles, reference)``.

    :param angles: Array of angles in radians.
    :param reference: Reference angle in radians.
    :returns: Array of signed differences in radians.
    """
    d = angles - reference
    return np.arctan2(np.sin(d), np.cos(d))


def _rotate_points(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    """
    Rotate 2-D points around a center.

    Matches MATLAB ``ctra_rotz(points - center, angle) + center``.

    :param points: (N, 2) array of [x, y] coordinates.
    :param angle: Rotation angle in radians.
    :param center: (2,) center of rotation [x, y].
    :returns: (N, 2) rotated points.
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (points - center) @ R.T + center


def classify_congruent_cells(
    result: ComparisonResult,
    params: ComparisonParams,
    reference_center: np.ndarray,
) -> None:
    """
    Identify Congruent Matching Cells (CMCs) using median procedure 6.

    This implements the MATLAB ``cell_cmc_median`` procedure 6:

    1. Compute per-cell registration angle difference.
    2. Take circular median as initial consensus angle.
    3. Apply ESD outlier rejection on angle residuals.
    4. Recompute median from inliers; tighten to 2 × angle_threshold; recompute.
    5. Rotate reference positions by consensus angle around reference_center.
    6. Compute position residuals; take median as consensus translation.
    7. Label cells within thresholds as congruent.

    :param result: ComparisonResult containing the list of cell results.
    :param params: Algorithm parameters with thresholds.
    :param reference_center: The global center [x, y] of the reference surface
        in micrometers, used as the center of rotation.  This corresponds to
        ``Map1.vCenterG`` in the MATLAB code.
    """
    cells = result.cells
    if not cells:
        return

    # --- Extract per-cell angles and positions ---
    # In MATLAB: angle1 = 0 for all cells (reference orientation),
    #            angle2 = registration_angle found during cell comparison.
    # The angle difference is therefore just registration_angle.
    angles = np.array([c.registration_angle for c in cells])  # radians
    pos_ref = np.array([c.center_reference for c in cells])  # (N, 2) in µm
    pos_comp = np.array([c.center_comparison for c in cells])  # (N, 2) in µm
    scores = np.array([c.area_cross_correlation_function_score for c in cells])

    # Keep track of which cells have valid (non-NaN) data
    valid = ~np.isnan(angles)

    if not np.any(valid):
        return

    # --- Step 1: Initial median angle ---
    angle_diffs = angles.copy()  # angle2 - angle1 where angle1 = 0
    consensus_angle = _circular_median(angle_diffs[valid])
    angle_residuals = _wrapped_angle_diff(angle_diffs, consensus_angle)

    # --- Step 2: ESD outlier rejection on angles ---
    valid_residuals = angle_residuals[valid]
    max_outliers = max(np.sum(valid) - 4, 0)
    outlier_mask_sub = detect_outliers_esd(
        valid_residuals, alpha=0.05, max_outliers=max_outliers
    )

    # Map back to full array: inliers among valid cells
    inlier_full = valid.copy()
    valid_indices = np.where(valid)[0]
    for i, idx in enumerate(valid_indices):
        if outlier_mask_sub[i]:
            inlier_full[idx] = False

    if np.any(inlier_full):
        # Recompute median from inliers
        consensus_angle = _circular_median(angle_diffs[inlier_full])
        angle_residuals = _wrapped_angle_diff(angle_diffs, consensus_angle)

        # Tighten: keep only cells within 2 * angle_threshold
        angle_threshold_rad = np.radians(params.angle_threshold)
        tight_mask = inlier_full & (np.abs(angle_residuals) <= 2 * angle_threshold_rad)

        if np.any(tight_mask):
            consensus_angle = _circular_median(angle_diffs[tight_mask])
            angle_residuals = _wrapped_angle_diff(angle_diffs, consensus_angle)

        # NaN-out rejected cells (matching MATLAB behaviour)
        rejected = valid & ~inlier_full
        angles[rejected] = np.nan
        pos_ref[rejected] = np.nan
        pos_comp[rejected] = np.nan
        scores[rejected] = np.nan
        angle_residuals[rejected] = np.nan

    # --- Step 3: Rotate reference positions and compute position residuals ---
    # p2_expected = R(consensus_angle) * (p1 - center) + center
    expected_pos = _rotate_points(pos_ref, consensus_angle, reference_center)

    # Position residuals: actual_comp - expected
    pos_residuals = pos_comp - expected_pos

    # Consensus translation = median of residuals
    consensus_translation = np.nanmedian(pos_residuals, axis=0)

    # Residuals relative to consensus
    pos_errors = pos_residuals - consensus_translation

    # --- Step 4: Label CMCs ---
    # MATLAB uses per-axis position check: |dx| <= distMax AND |dy| <= distMax
    angle_threshold_rad = np.radians(params.angle_threshold)

    for i, cell in enumerate(cells):
        cell.is_congruent = bool(
            scores[i] >= params.correlation_threshold
            and not np.isnan(angle_residuals[i])
            and np.abs(angle_residuals[i]) <= angle_threshold_rad
            and np.abs(pos_errors[i, 0]) <= params.position_threshold
            and np.abs(pos_errors[i, 1]) <= params.position_threshold
        )

    # --- Store consensus ---
    result.consensus_rotation = float(consensus_angle)
    result.consensus_translation = consensus_translation
    result.update_summary()
