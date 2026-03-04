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


def _get_esd_criterion(values: FloatArray1D) -> BoolArray1D:
    max_outliers = max(len(values) - 4, 0)
    outliers_mask = _outliers_gesd(data=values, outliers=max_outliers, alpha=0.05)
    return ~outliers_mask


def _get_threshold_criterion(values: FloatArray1D, threshold: float) -> BoolArray1D:
    return np.abs(values) <= 2 * threshold


def _get_consensus_angle(cells: list[Cell], threshold: float) -> float:
    """TODO ESD outlier rejection on angles ---."""
    angles = np.radians([c.angle_deg for c in cells])

    # Compute median from angles
    consensus_angle = _circular_median(angles=angles)
    angle_residuals = _wrapped_angle_diff(angles=angles, reference=consensus_angle)
    mask = _get_esd_criterion(values=angle_residuals)
    if not np.any(mask):
        raise RuntimeError("All cells are outliers.")

    # Recompute median based on the inliers
    consensus_angle = _circular_median(angles[mask])
    angle_residuals = _wrapped_angle_diff(angles=angles, reference=consensus_angle)

    # Tighten: re-evaluate all cells against 2 × angle_threshold
    mask = _get_threshold_criterion(values=angle_residuals, threshold=threshold)
    if np.any(mask):
        consensus_angle = _circular_median(angles[mask])
        angle_residuals = _wrapped_angle_diff(angles=angles, reference=consensus_angle)

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
        cell.is_congruent = (
            cell.best_score >= params.correlation_threshold
            and not cell.meta_data.is_outlier
            and np.abs(cell.meta_data.residual_angle_deg) <= params.angle_threshold
            and np.all(
                np.abs(cell.meta_data.position_error) <= params.position_threshold
            )
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


def _outliers_gesd(data: np.ndarray, outliers: int, alpha: float) -> np.ndarray:
    """
    Generalised ESD test for outliers (Rosner 1983), matching the MATLAB
    ``stat_idout_esd`` reference implementation used by the NIST CMC pipeline.

    Critical values use the Rosner formula:
        lambda_i = t(p, ni-1) * ni / sqrt((ni-1 + t^2) * (ni+1))
        where p = 1 - alpha / (2*(ni+1)) and ni = remaining observations
        AFTER the i-th removal (i.e. n - i).

    :param data: 1-D array of values to test.
    :param outliers: Maximum number of outliers to test for.
    :param alpha: Significance level.
    :returns: Boolean mask in original array order; True = outlier.
    """
    n = len(data)
    if outliers <= 0 or n < 3:
        return np.zeros(n, dtype=bool)  # All inliers

    argsort_index = np.argsort(data)
    data_sorted = data[argsort_index]

    # First pass: compute R statistics and critical values, tracking which
    # sorted-array position is removed at each step.
    r_statistics: list[float] = []  # r_statistic is max(abs(z_scores)). test_statistic.
    lambdas: list[float] = []  # test critical values
    data_work = list(
        data_sorted
    )  # convert to list since we need to pop elements later.
    remaining_positions = list(range(n))  # sorted-array positions still in data_work

    for _ in range(outliers):
        if len(data_work) < 3:
            break
        arr = np.array(data_work)
        mean_work = float(np.mean(arr))
        std_work = float(np.std(arr, ddof=1))
        if std_work == 0:
            break
        idx = int(np.argmax(np.abs(arr - mean_work)))  # idx with max absolute z_value
        r_statistics.append(
            float(np.abs(arr[idx] - mean_work) / std_work)
        )  # store this value

        # Critical value uses ni = remaining count AFTER this removal
        ni = len(data_work) - 1
        if ni >= 2:
            p = 1.0 - alpha / (2.0 * (ni + 1))
            rt = float(t.ppf(p, ni - 1))
            lambdas.append(
                rt * ni / np.sqrt((ni - 1 + rt**2) * (ni + 1))
            )  # this is the critical value formula used by (Rosner 1983)
        else:
            lambdas.append(np.inf)

        remaining_positions.pop(idx)
        data_work.pop(idx)

    # Find the last step i (1-indexed) where H0 is rejected
    last_reject = 0
    for i, (r_statistic, critical_value) in enumerate(zip(r_statistics, lambdas)):
        if r_statistic > critical_value:
            last_reject = i + 1

    if last_reject == 0:
        return np.zeros(n, dtype=bool)

    # Second pass: retrace the first `last_reject` removals to identify
    # which sorted-array positions are outliers.
    data_work2 = list(data_sorted)
    remaining2 = list(range(n))
    outlier_sorted_idxs: list[int] = []
    for _ in range(last_reject):
        arr = np.array(data_work2)
        idx = int(np.argmax(np.abs(arr - np.mean(arr))))
        outlier_sorted_idxs.append(remaining2[idx])
        remaining2.pop(idx)
        data_work2.pop(idx)

    mask_sorted = np.zeros(n, dtype=bool)
    for idx in outlier_sorted_idxs:
        mask_sorted[idx] = True

    # Unsort back to original input order
    mask_original = np.zeros(n, dtype=bool)
    mask_original[argsort_index] = mask_sorted
    return mask_original


def _circular_median(angles: FloatArray1D) -> float:
    """
    Compute the circular median of a set of angles (in radians).

    The circular median minimizes the sum of absolute angular distances.

    :param angles: 1-D array of angles in radians.
    :returns: The circular median angle in radians.
    """
    best_idx = 0
    best_cost = np.inf
    for i, candidate in enumerate(angles):
        raw_diff = angles - candidate
        wrapped_diff = (raw_diff + np.pi) % (2 * np.pi) - np.pi
        cost = np.sum(np.abs(wrapped_diff))
        if cost < best_cost:
            best_cost = cost
            best_idx = i

    ref = angles[best_idx]
    centred = (angles - ref + np.pi) % (2 * np.pi) - np.pi
    med = float(np.median(centred))
    result = (ref + med + np.pi) % (2 * np.pi) - np.pi
    return float(result)


def _wrapped_angle_diff(angles: FloatArray1D, reference: float) -> FloatArray1D:
    """
    Signed angular difference wrapped to [-pi, pi].

    :param angles: Array of angles in radians.
    :param reference: Reference angle in radians.
    :returns: Array of signed differences in radians.
    """
    d = angles - reference
    return np.arctan2(np.sin(d), np.cos(d))


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
