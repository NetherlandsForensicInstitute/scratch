import numpy as np
from scipy.stats import t

from container_models.base import (
    Points2D,
    Point2D,
    FloatArray1D,
    BoolArray1D,
)
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)


def classify_congruent_cells(
    cells: list[Cell],
    params: ComparisonParams,
    reference_center: np.ndarray,
) -> ComparisonResult:
    """
    Identify Congruent Matching Cells (CMCs) using median procedure with ESD outlier rejection for angles.

    Steps:
    TODO: explain algorithmic steps here

    :param cells: The list of computed cells.
    :param params: Algorithm parameters with thresholds.
    :param reference_center: The global center [x, y] of the reference surface in meters,
        used as the center of rotation.
    """
    if not cells:
        raise ValueError("Cannot classify CMC from an empty list.")

    consensus_angle = _get_consensus_angle(
        cells=cells, threshold=np.radians(params.angle_threshold)
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


def _rotate_points(points: Points2D, angle: float, center: Point2D) -> Points2D:
    """
    Rotate 2-D points around a center.

    :param points: (N, 2) array of [x, y] coordinates.
    :param angle: Rotation angle in radians.
    :param center: (2,) array for the center of rotation [x, y].
    :returns: (N, 2) rotated points.
    """
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
    return (points - center) @ rotation_matrix.T + center


def _get_esd_criterion(values: FloatArray1D) -> BoolArray1D:
    max_outliers = max(len(values) - 4, 0)
    outliers_mask = _outliers_gesd(data=values, max_outliers=max_outliers, alpha=0.05)
    return ~outliers_mask


def _get_threshold_criterion(values: FloatArray1D, threshold: float) -> BoolArray1D:
    return np.abs(values) <= 2 * threshold


def _get_consensus_angle(cells: list[Cell], threshold: float) -> float:
    """TODO ESD outlier rejection on angles ---."""
    angles = np.radians([c.angle_deg for c in cells])

    # Compute median from angles
    consensus_angle = _circular_median(angles=angles)
    angle_residuals = _wrap_angles(angles=angles - consensus_angle)
    mask = _get_esd_criterion(values=angle_residuals)
    if not np.any(mask):
        raise RuntimeError("All cells are outliers.")

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
    cells: list[Cell], angle: float, rotation_center: Point2D
) -> Point2D:
    """TODO: Rotate reference positions and compute position residuals ---."""
    centers_reference = np.array([c.center_reference for c in cells])
    centers_comparison = np.array([c.center_comparison for c in cells])
    outliers = np.array([c.meta_data.is_outlier for c in cells])
    centers_reference[outliers] = np.nan
    centers_comparison[outliers] = np.nan
    expected_positions_on_reference = _rotate_points(
        points=centers_reference, angle=angle, center=rotation_center
    )
    # Compute residuals with respect to comparison.
    position_residuals = centers_comparison - expected_positions_on_reference
    consensus_translation = np.nanmedian(position_residuals, axis=0)
    position_errors = position_residuals - consensus_translation

    # Update cell meta-data
    for cell, position_error in zip(cells, position_errors):
        cell.meta_data.position_error = position_error

    return consensus_translation


def _update_congruence(cells: list[Cell], params: ComparisonParams):
    for cell in cells:
        cell.is_congruent = bool(
            cell.best_score >= params.correlation_threshold
            and not cell.meta_data.is_outlier
            and np.abs(cell.meta_data.residual_angle_deg) <= params.angle_threshold
            and np.all(
                np.abs(cell.meta_data.position_error) <= params.position_threshold
            )
        )


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
