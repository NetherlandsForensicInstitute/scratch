import numpy as np
from scipy.stats import t

from container_models.base import FloatArray1D, BoolArray1D
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)
from conversion.surface_comparison.utils import rotate_points


def classify_congruent_cells(
    cells: list[Cell],
    params: ComparisonParams,
    reference_center: tuple[float, float],
) -> ComparisonResult:
    """
    Identify Congruent Matching Cells (CMCs) using a median-based procedure with
    generalized ESD outlier rejection.

    Steps:
    1. **Consensus angle** — compute a circular median of all cell registration
       angles, reject statistical outliers via the generalized ESD test, tighten
       the inlier set to cells within ``2 × angle_threshold``, and recompute the
       median. Each cell's ``meta_data.is_outlier`` and
       ``meta_data.residual_angle_deg`` are set accordingly.
    2. **Consensus translation** — rotate every reference cell center by the
       consensus angle, then take the component-wise median of the offsets between
       rotated reference centers and comparison centres, excluding outlier cells.
       Each cell's ``meta_data.position_error`` is set to its deviation from this
       consensus.
    3. **Congruence labeling** — mark each cell as congruent when it meets all
       four criteria: correlation score ≥ threshold, not an angle outlier,
       ``|residual_angle_deg| ≤ angle_threshold``, and both position error
       components within ``position_threshold``.

    :param cells: Per-cell registration results to classify.
    :param params: Algorithm parameters (thresholds for score, angle, and position).
    :param reference_center: Global center [x, y] of the reference surface in meters,
        used as the fixed point for the consensus rotation.
    :returns: A :class:`ComparisonResult` containing the classified cells, consensus
        rotation in degrees, and consensus translation in meters.
    :raises ValueError: If ``cells`` is empty.
    :raises RuntimeError: If the ESD test rejects every cell as an angle outlier.
    """
    if not cells:
        raise ValueError("Cannot identify CMC from an empty list.")

    consensus_angle = _get_consensus_angle(
        cells=cells, threshold=np.radians(params.angle_deviation_threshold)
    )
    # Rotate reference positions to compute the translation.
    # Since we rotate the reference image, the natural center for rotation
    # is the mid of the reference image defined in reference center.
    consensus_translation = _get_consensus_translation(
        cells=cells, angle=consensus_angle, rotation_center=reference_center
    )
    _update_congruence(cells=cells, params=params)

    result = ComparisonResult(
        cells=cells,
        consensus_rotation=float(np.degrees(consensus_angle)),
        consensus_translation=consensus_translation,
    )
    return result


def _circular_median(angles: FloatArray1D) -> float:
    """
    Compute the circular median of a set of angles (in radians).

    The circular median minimizes the sum of absolute angular distances.

    :param angles: 1-D array of angles in radians.
    :returns: The circular median angle in radians.
    """
    costs = [np.sum(np.abs(_wrap_angles(angles - a))) for a in angles]
    ref = angles[np.argmin(costs)]
    return float(_wrap_angles(ref + np.median(_wrap_angles(angles - ref))))


def _wrap_angles(angles: FloatArray1D) -> FloatArray1D:
    """
    Normalize angles in radians to the [-pi, pi] interval.

    :param angles: Array of angles in radians.
    :returns: Array of normalized angles in radians.
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _get_esd_criterion(values: FloatArray1D) -> BoolArray1D:
    """
    Return a boolean inlier mask for ``values`` using the generalized ESD test.

    The maximum number of outliers tested is ``len(values) - 4``, preserving at
    least four observations for a stable median estimate.

    :param values: 1-D array of residuals to screen.
    :returns: Boolean mask; True = inlier, False = outlier.
    """
    max_outliers = max(len(values) - 4, 0)
    outliers_mask = _outliers_gesd(data=values, max_outliers=max_outliers, alpha=0.05)
    return ~outliers_mask


def _get_threshold_criterion(values: FloatArray1D, threshold: float) -> BoolArray1D:
    """
    Return a boolean inlier mask where ``|value| ≤ 2 * threshold``.

    The factor of two provides a looser tightening gate in the first pass of
    consensus-angle estimation before the final single-threshold classification.

    :param values: 1-D array of residuals to screen.
    :param threshold: Half-width of the acceptance band.
    :returns: Boolean mask; True = inlier, False = outlier.
    """
    return np.abs(values) <= 2 * threshold


def _get_consensus_angle(cells: list[Cell], threshold: float) -> float:
    """
    Estimate the consensus rotation angle across all cells using a three-step
    median-and-rejection procedure.

    1. Compute an initial circular median of all cell angles.
    2. Apply the generalized ESD test to the angle residuals to remove statistical
       outliers, then recompute the median from inliers.
    3. Tighten the inlier set to cells within ``2 × threshold`` of the new median
       and recompute once more.

    Each cell's ``meta_data.is_outlier`` and ``meta_data.residual_angle_deg`` are
    updated to reflect the final inlier/outlier decision.

    :param cells: List of cells providing ``angle_deg`` measurements.
    :param threshold: Half-width acceptance band in radians (typically the
        ``angle_threshold`` parameter converted from degrees).
    :returns: Consensus rotation angle in radians.
    :raises RuntimeError: If the ESD test rejects every cell.
    """
    angles = np.radians([c.angle_deg for c in cells])

    # Compute median from angles
    consensus_angle = _circular_median(angles=angles)
    angle_residuals = _wrap_angles(angles=angles - consensus_angle)
    mask = _get_esd_criterion(values=angle_residuals)

    # Recompute median based on the inliers
    consensus_angle = _circular_median(angles[mask])
    angle_residuals = _wrap_angles(angles=angles - consensus_angle)

    # Tighten: re-evaluate all cells against 2 × angle_threshold
    mask = _get_threshold_criterion(values=angle_residuals, threshold=threshold)
    if np.any(mask):
        consensus_angle = _circular_median(angles[mask])
        angle_residuals = _wrap_angles(angles=angles - consensus_angle)

    # Update cell meta-data
    for cell, is_outlier, residual_angle in zip(cells, ~mask, angle_residuals):
        cell.meta_data.is_outlier = bool(is_outlier)
        cell.meta_data.residual_angle_deg = float(np.degrees(residual_angle))

    return consensus_angle


def _get_consensus_translation(
    cells: list[Cell], angle: float, rotation_center: tuple[float, float]
) -> tuple[float, float]:
    """
    Rotate reference cell centers by the consensus angle, then compute the
    median positional offset between the rotated reference and comparison centers.

    Outlier cells are excluded from the median by NaN-masking their centers before aggregation.
    Every cell's ``meta_data.position_error`` is updated with its signed [x, y] deviation
    from the consensus translation.

    :param cells: List of cells whose ``meta_data.is_outlier`` flags are already set.
    :param angle: Consensus rotation angle in radians.
    :param rotation_center: [x, y] center of rotation in meters.
    :returns: Consensus translation [x, y] in meters.
    """
    centers_reference = np.array([c.center_reference for c in cells])
    centers_comparison = np.array([c.center_comparison for c in cells])
    outliers = np.array([c.meta_data.is_outlier for c in cells])
    centers_reference[outliers] = np.nan
    centers_comparison[outliers] = np.nan
    centers_reference_temp = centers_reference.copy()
    centers_reference_temp[:, 1] *= -1
    rotation_center_temp = list(rotation_center)
    rotation_center_temp[1] *= -1
    expected_positions_in_comparison_frame = rotate_points(
        points=centers_reference_temp, angle=angle, center=tuple(rotation_center_temp)
    )
    expected_positions_in_comparison_frame[:, 1] *= -1
    # Compute residuals with respect to comparison.
    position_residuals = centers_comparison - expected_positions_in_comparison_frame
    consensus_translation = np.nanmedian(position_residuals, axis=0)
    position_errors = position_residuals - consensus_translation

    # Update cell meta-data
    for cell, position_error in zip(cells, position_errors):
        cell.meta_data.position_error = (
            float(position_error[0]),
            float(position_error[1]),
        )

    return float(consensus_translation[0]), float(consensus_translation[1])


def _update_congruence(cells: list[Cell], params: ComparisonParams) -> None:
    """
    Set ``is_congruent`` on each cell based on all four CMC criteria.

    A cell is congruent when it satisfies every condition simultaneously:
    - ``best_score ≥ correlation_threshold`` — sufficient cross-correlation quality.
    - ``not meta_data.is_outlier`` — not rejected as an angle outlier by ESD or
      the tightening step.
    - ``|meta_data.residual_angle_deg| ≤ angle_threshold`` — angular deviation
      from the consensus rotation is within tolerance.
    - ``|meta_data.position_error[x]| ≤ position_threshold`` and
      ``|meta_data.position_error[y]| ≤ position_threshold`` — positional deviation
      from the consensus translation is within tolerance in both axes.

    :param cells: Cells whose ``meta_data`` fields have already been populated by
        :func:`_get_consensus_angle` and :func:`_get_consensus_translation`.
    :param params: Algorithm parameters providing the classification thresholds.
    """
    for i, cell in enumerate(cells):
        congruent = bool(
            cell.best_score >= params.correlation_threshold
            and not cell.meta_data.is_outlier
            and np.abs(cell.meta_data.residual_angle_deg)
            <= params.angle_deviation_threshold
            and np.all(
                np.abs(cell.meta_data.position_error) <= params.position_threshold
            )
        )
        cell.is_congruent = congruent


def _rosner_critical_value(n_remaining: int, alpha: float) -> float:
    """
    Rosner (1983) critical value for the GESD test.

    :param n_remaining: Number of observations after removal.
    :param alpha: Significance level.
    :returns: Critical value lambda_i.
    """
    if n_remaining < 2:
        return np.inf
    p = 1.0 - alpha / (2.0 * (n_remaining + 1))
    t_value = float(t.ppf(p, n_remaining - 1))
    return (
        t_value
        * n_remaining
        / np.sqrt((n_remaining - 1 + t_value**2) * (n_remaining + 1))
    )


def _outliers_gesd(
    data: FloatArray1D, max_outliers: int, alpha: float = 0.05
) -> BoolArray1D:
    """
    Generalised ESD test for outliers (Rosner 1983), matching the MATLAB
    ``stat_idout_esd`` reference implementation used by the NIST CMC pipeline.

    Iteratively removes the most extreme value and tests whether it is a
    statistically significant outlier. The final outlier count is the last
    iteration where the test statistic exceeds the critical value, which
    handles masking effects where removing one outlier reveals another.

    Tie-breaking: when multiple values share the maximum deviation from the
    mean, the one with the lowest array index is removed first.

    :param data: 1-D array of values to test.
    :param max_outliers: Maximum number of outliers to test for.
    :param alpha: Significance level.
    :returns: Boolean mask in original array order; True = outlier.
    """
    n = len(data)
    if max_outliers <= 0 or n < 3:
        return np.zeros(n, dtype=bool)

    remaining = np.arange(n)
    values = data.copy()
    removed_indices: list[int] = []
    last_reject = 0

    for i in range(max_outliers):
        if len(values) < 3:
            break

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if std == 0:
            break

        deviations = np.abs(values - mean)
        idx = int(np.argmax(deviations))
        r_statistic = deviations[idx] / std
        critical_value = _rosner_critical_value(len(values) - 1, alpha)

        removed_indices.append(remaining[idx])

        if r_statistic > critical_value:
            last_reject = i + 1

        values = np.delete(values, idx)
        remaining = np.delete(remaining, idx)

    mask = np.zeros(n, dtype=bool)
    mask[removed_indices[:last_reject]] = True
    return mask
