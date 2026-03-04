
import numpy as np
from scipy.stats import t
from collections.abc import Sequence
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)

Point2D = tuple[float, float]  # (X, Y) coordinates


def classify_congruent_cells(
    cells: Sequence[Cell],
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

    angle_threshold_rad = np.radians(params.angle_threshold)

    angles = np.radians([c.angle_reference for c in cells])
    centers_references = np.array([c.center_reference for c in cells])  # (N, 2) in m
    centers_comparisons = np.array([c.center_comparison for c in cells])  # (N, 2) in m
    scores = np.array([c.best_score for c in cells])

    # --- Step 1: Initial median angle ---
    consensus_angle = _circular_median(angles)
    angle_residuals = _wrapped_angle_diff(angles=angles, reference=consensus_angle)

    # --- Step 2: ESD outlier rejection on angles ---
    max_outliers = max(len(angles) - 4, 0)
    outlier_mask_sub = _outliers_gesd(
        angle_residuals, outliers=max_outliers, alpha=0.05
    )
    inliers = ~outlier_mask_sub
    if not np.any(inliers):
        raise RuntimeError("All cells are outlier.")

    # Recompute median from inliers
    consensus_angle = _circular_median(angles[inliers])
    angle_residuals = _wrapped_angle_diff(angles=angles, reference=consensus_angle)

    # Tighten: re-evaluate all cells against 2 × angle_threshold
    inliers = np.abs(angle_residuals) <= 2 * angle_threshold_rad  # overwrite `inliers`

    if np.any(inliers):
        consensus_angle = _circular_median(angles[inliers])
        angle_residuals = _wrapped_angle_diff(angles, consensus_angle)

    # NaN-out rejected cells (ESD + tightening)
    rejected = ~inliers
    angles[rejected] = np.nan
    centers_references[rejected] = np.nan
    centers_comparisons[rejected] = np.nan
    scores[rejected] = np.nan
    angle_residuals[rejected] = np.nan

    # --- Step 3: Rotate reference positions and compute position residuals ---
    # Since we rotate the reference image the natural center for rotation is the mid of the reference image defined in
    # reference center.
    expected_positions_on_reference = _rotate_points(
        centers_references, consensus_angle, reference_center
    )
    position_residuals = (
        centers_comparisons - expected_positions_on_reference
    )  # Comparison = base. Residuals with respect to comparison.
    consensus_translation = np.nanmedian(position_residuals, axis=0)
    position_errors = position_residuals - consensus_translation

    # --- Step 4: Label CMCs ---
    for i, cell in enumerate(cells):
        cell.is_congruent = bool(
            scores[i] >= params.correlation_threshold
            and not np.isnan(angle_residuals[i])
            and np.abs(angle_residuals[i]) <= angle_threshold_rad
            and np.abs(position_errors[i, 0]) <= params.position_threshold
            and np.abs(position_errors[i, 1]) <= params.position_threshold
        )

    result = ComparisonResult(
        cells=cells,
        consensus_rotation=float(consensus_angle) * 180 / np.pi,
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


def _circular_median(angles: np.ndarray) -> float:
    """
    Compute the circular median of a set of angles (in radians).

    The circular median minimises the sum of absolute angular distances.

    :param angles: 1-D array of angles in radians.
    :returns: The circular median angle in radians.
    """
    angles = angles[~np.isnan(angles)]
    if angles.size == 0:
        return np.nan

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


def _wrapped_angle_diff(angles: np.ndarray, reference: float) -> np.ndarray:
    """
    Signed angular difference wrapped to [-pi, pi].

    :param angles: Array of angles in radians.
    :param reference: Reference angle in radians.
    :returns: Array of signed differences in radians.
    """
    d = angles - reference
    return np.arctan2(np.sin(d), np.cos(d))


def _rotate_points(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    """
    Rotate 2-D points around a center.

    :param points: (N, 2) array of [x, y] coordinates.
    :param angle: Rotation angle in radians.
    :param center: (2,) center of rotation [x, y].
    :returns: (N, 2) rotated points.
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (points - center) @ R.T + center


#
# def _rotate_point(point: Point2D, angle: float, center: Point2D) -> Point2D:
#     """Clock-wise rotate a 2D point (px, py) around a center (cx, cy) by `angle` radians."""
#     translated = (point[0] - center[0], point[1] - center[1])
#     rotated_x = math.cos(angle) * translated[0] - math.sin(angle) * translated[1]
#     rotated_y = math.sin(angle) * translated[0] + math.cos(angle) * translated[1]
#     return rotated_x, rotated_y
