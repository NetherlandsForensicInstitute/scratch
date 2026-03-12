import numpy as np
from scipy.stats import t

from container_models.base import Points2D, FloatArray1D, BoolArray1D
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)


def classify_congruent_cells_consensus(
    cells: list[Cell],
    params: ComparisonParams,
    reference_center: tuple[float, float],
) -> ComparisonResult:
    """
    Identify Congruent Matching Cells (CMCs) using a median-based procedure with
    generalized ESD outlier rejection.

    Steps:
    1. Filter cells that pass the similarity threshold.
    2. Loop over all pairs (i,j) of those cells, and for each pair:
       Estimate a rigid body transformation (rotation + translation) from just those two cells via _get_cmc_consensus → _find_consensus_parameters
       Find all other cells that fall within position_threshold and angle_deviation_threshold of that predicted location.
       Attempt to iteratively refine by re-fitting using all successful cells (a least-squares improvement step).
       Keep the solution if it yields more CMC cells than the current best (or equal count with better quality).
    3. Get a boolean vector flagging which cells are CMC.
    4. Return a ComparisonResult.

    :param cells: Per-cell registration results to classify.
    :param params: Algorithm parameters (thresholds for score, angle, and position).
    :param reference_center: Global center [x, y] of the reference surface in meters,
        used as the fixed point for the median rotation.
    :returns: A `ComparisonResult` containing the classified cells, consensus
        rotation in degrees, and consensus translation in meters.
    :raises ValueError: If ``cells`` is empty.
    """
    if not cells:
        raise ValueError("Cannot identify CMC from an empty list.")

    filtered_cells = _filter_cells(cells, params.minimum_fill_fraction)
    n_filtered_cells = len(filtered_cells)

    max_distance = params.position_threshold          # in meters
    max_abs_angle_distance = params.angle_deviation_threshold  # in degrees

    # initialize solution: default to first cell as sole CMC
    best_inliers_idx: list[int] = [0]
    criterion = np.inf

    if n_filtered_cells <= 1 or (np.isinf(max_distance) and max_abs_angle_distance == 180):
        # Zero, one, or all cells are CMC — skip pair search
        best_inliers_idx = list(range(n_filtered_cells))
    else:
        for idx_1 in range(n_filtered_cells):
            for idx_2 in range(idx_1 + 1, n_filtered_cells):

                # --- Seed: evaluate two-cell pair solution ---
                cell_distances, cell_angle_distances = _get_cmc_consensus(
                    [idx_1, idx_2], filtered_cells, reference_center
                )
                inliers_idx_new = np.where(
                    (cell_distances <= max_distance)
                    & (cell_angle_distances <= max_abs_angle_distance)
                )[0].tolist()

                criterion_current = _calculate_criterion(
                    inliers_idx_new, cell_distances, cell_angle_distances,
                    max_distance, max_abs_angle_distance,
                )

                if 2 < len(inliers_idx_new) < n_filtered_cells:
                    # --- Refinement: iteratively re-fit using all inliers ---
                    inliers_idx_prev_inner = inliers_idx_new

                    while len(inliers_idx_new) >= len(inliers_idx_prev_inner):
                        cell_distances, cell_angle_distances = _get_cmc_consensus(
                            inliers_idx_new, filtered_cells, reference_center
                        )
                        inliers_idx_candidate = np.where(
                            (cell_distances <= max_distance)
                            & (cell_angle_distances <= max_abs_angle_distance)
                        )[0].tolist()

                        criterion_new = _calculate_criterion(
                            inliers_idx_candidate, cell_distances, cell_angle_distances,
                            max_distance, max_abs_angle_distance,
                        )

                        # Accept if strictly more inliers, or same count with lower criterion
                        if len(inliers_idx_candidate) > len(inliers_idx_new) or (
                            len(inliers_idx_candidate) == len(inliers_idx_new)
                            and criterion_new < criterion_current
                        ):
                            criterion_current = criterion_new
                            inliers_idx_prev_inner = inliers_idx_new
                            inliers_idx_new = inliers_idx_candidate
                        else:
                            # Stop trying to improve
                            break

                # --- Accept global best if this pair solution is better ---
                if len(inliers_idx_new) > len(best_inliers_idx) or (
                    len(inliers_idx_new) == len(best_inliers_idx)
                    and criterion_current < criterion
                ):
                    best_inliers_idx = inliers_idx_new
                    criterion = criterion_current

            if len(best_inliers_idx) == n_filtered_cells:
                break  # outer loop short-circuit

    # --- Pick the single highest-scoring cell if no pair improved things ---
    if len(best_inliers_idx) == 1:
        best_scores = [filtered_cells[i].best_score for i in range(n_filtered_cells)]
        best_inliers_idx = [int(np.argmax(best_scores))]

    # --- Mark cells as congruent ---
    congruent_set = set(
        filtered_cells[i] for i in best_inliers_idx
    )
    # We need original cell indices; map filtered back via identity
    updated_cells = []
    for cell in cells:
        is_congruent = cell in congruent_set
        updated_cells.append(cell.model_copy(update={"is_congruent": is_congruent}))

    # --- Estimate shared transformation from CMC inliers ---
    if len(best_inliers_idx) > 1:
        cmc_cells = [filtered_cells[i] for i in best_inliers_idx]
        shared_translation, shared_rotation_rad = _find_consensus_parameters(
            cmc_cells, reference_center
        )
        shared_rotation_deg = float(np.degrees(shared_rotation_rad))
    else:
        shared_translation = (float("nan"), float("nan"))
        shared_rotation_deg = float("nan")

    return ComparisonResult(
        cells=updated_cells,
        shared_rotation=shared_rotation_deg,
        shared_translation=shared_translation,
    )


def _filter_cells(
    cells: list[Cell], minimum_fill_fraction: float
) -> list[Cell]:
    """Keep cells that have fill_fraction_reference >= minimum_fill_fraction.

    :param cells: a list of Cells to filter
    :param minimum_fill_fraction: minimum fill fraction of filtered cells
    :returns: filtered_cells, a list of filtered cells
    """
    filtered_cells = [
        cell for cell in cells
        if cell.fill_fraction_reference >= minimum_fill_fraction
    ]
    return filtered_cells


def _calculate_criterion(
    cell_ids: list[int],
    cell_distances: np.ndarray,
    cell_angle_distances: np.ndarray,
    max_distance: float,
    max_abs_angle_distance: float,
) -> float:
    """Calculate criterion: mean_normalized_distance + mean_normalized_angle_distance.

    :param cell_ids: a list of cell IDs (indices into cell_distances / cell_angle_distances)
    :param cell_distances: array of distances between cells and predicted positions (meters)
    :param cell_angle_distances: array of absolute angle distances (degrees)
    :param max_distance: maximum distance threshold (meters)
    :param max_abs_angle_distance: maximum absolute angle threshold (degrees)
    :returns: criterion (float)
    """
    if not cell_ids:
        return np.inf
    criterion = (
        float(np.mean(cell_distances[cell_ids])) / max_distance
        + float(np.mean(cell_angle_distances[cell_ids])) / max_abs_angle_distance
    )
    return criterion


def _get_cmc_consensus(
    included_idx: list[int],
    cells: list[Cell],
    reference_center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate cell_distances and cell_angle_distances for all cells after finding consensus parameters.

    :param included_idx: a list of included cell indices (used for least-squares fit)
    :param cells: a list of all filtered cells
    :param reference_center: global center [x, y] used as the fixed rotation point (meters)
    :returns: cell_distances (meters), cell_angle_distances in absolute degrees — both as np.ndarray of length len(cells)
    """
    cells_for_least_squares = [cells[idx] for idx in included_idx]
    consensus_translation, consensus_rotation_rad = _find_consensus_parameters(
        cells_for_least_squares, reference_center
    )
    cell_distances, cell_angle_distances = _get_distances(
        cells, consensus_translation, consensus_rotation_rad
    )
    return cell_distances, cell_angle_distances


def _find_consensus_parameters(
    cells: list[Cell],
    reference_center: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """Least-squares fit to cells' rotation and translation to find consensus parameters.

    Replicates the MATLAB ``point_register`` function (SVD-based Procrustes):
      - Centres both point clouds around their respective means (or vCenter when
        translation is fixed).
      - Solves for the best-fit rotation via SVD of the cross-covariance matrix.
      - Computes the translation that maps the rotated mean of mPos1 to the mean of mPos2.

    The rotation angle in radians is recovered as ``atan2(-T[0,1], T[0,0])``,
    matching the MATLAB convention ``atan2(-mT(1,2), mT(1,1))``.

    :param cells: cells whose (center_reference, center_comparison) pairs are used for fitting
    :param reference_center: global center [x, y] used as the fixed rotation point (meters)
    :returns: consensus_translation (tx, ty) in meters,
              consensus_rotation_rad (float) in radians
    """
    pos1 = np.array([cell.center_reference for cell in cells], dtype=float)   # (N, 2)
    pos2 = np.array([cell.center_comparison for cell in cells], dtype=float)  # (N, 2)
    v_center = np.array(reference_center, dtype=float)                         # (2,)

    # --- Remove NaN rows ---
    valid = ~(np.any(np.isnan(pos1), axis=1) | np.any(np.isnan(pos2), axis=1))
    pos1 = pos1[valid]
    pos2 = pos2[valid]

    # --- Default outputs ---
    v_trans = np.zeros(2)
    rotation_matrix = np.eye(2)

    if len(pos1) == 0:
        return (float(v_trans[0]), float(v_trans[1])), 0.0

    # --- Centroids (vbRegister = [1,1,0,0]: estimate translation + rotation) ---
    vpos1 = pos1.mean(axis=0)   # mean of reference positions
    vpos2 = pos2.mean(axis=0)   # mean of comparison positions

    pos1_c = pos1 - vpos1       # centred reference positions
    pos2_c = pos2 - vpos2       # centred comparison positions

    # --- SVD for best-fit rotation (no reflection, no scaling) ---
    # M = pos1_c^T * pos2_c  (MATLAB: mPos1' * mPos2)
    M = pos1_c.T @ pos2_c       # (2, 2)
    U, _, Vt = np.linalg.svd(M)
    V = Vt.T
    rotation_matrix = V @ U.T   # (2, 2)

    # If det < 0 we have an unintended reflection; correct it
    # (vbRegister(4) == 0 means reflections are NOT allowed)
    if np.linalg.det(rotation_matrix) < 0:
        V[:, -1] *= -1
        rotation_matrix = V @ U.T

    # --- Translation (MATLAB: vTrans = vpos2 - vpos1 + rScale*(vCenter - vpos1)*mT' + (vpos1 - vCenter))
    # with rScale = 1 and no scaling:
    #   vTrans = vpos2 - vpos1 + (vCenter - vpos1) @ rotation_matrix.T + vpos1 - vCenter
    v_trans = (
        vpos2
        - vpos1
        + (v_center - vpos1) @ rotation_matrix.T
        + vpos1
        - v_center
    )

    # --- Rotation angle: atan2(-T[0,1], T[0,0])  (matches MATLAB atan2(-mT(1,2), mT(1,1))) ---
    consensus_rotation_rad = float(np.arctan2(-rotation_matrix[0, 1], rotation_matrix[0, 0]))
    consensus_translation = (float(v_trans[0]), float(v_trans[1]))

    return consensus_translation, consensus_rotation_rad


def _get_distances(
    cells: list[Cell],
    consensus_translation: tuple[float, float],
    consensus_rotation_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Get distances and abs_angle_distances of cell locations/rotations versus
    consensus_translation and consensus_rotation_rad.

    :param cells: a list of cells
    :param consensus_translation: (tx, ty) translation in meters (applied after rotation)
    :param consensus_rotation_rad: rotation in radians (used to predict comparison positions
        AND to compute angle residuals after converting to degrees)
    :returns: distances (meters) as np.ndarray, abs_angle_distances (unsigned degrees) as np.ndarray
    """
    predicted_positions = _predict_positions(cells, consensus_translation, consensus_rotation_rad)
    distances = np.array(_get_distances_meters(cells, predicted_positions))

    consensus_rotation_deg = float(np.degrees(consensus_rotation_rad))
    # angle_deg on each cell is already the angle difference (angle2 - angle1 in MATLAB).
    # The absolute residual is |cell.angle_deg - consensus_rotation_deg|.
    abs_angle_distances = np.array(
        [abs(cell.angle_deg - consensus_rotation_deg) for cell in cells]
    )

    return distances, abs_angle_distances


def _predict_positions(
    cells: list[Cell],
    consensus_translation: tuple[float, float],
    consensus_rotation_rad: float,
) -> list[tuple[float, float]]:
    """Predict positions of cells after rotation and translation by consensus values.

    Replicates MATLAB ``point_transform_2d``:
        mPosPred2 = rotate(mPos1 - vCenter, angle) + vCenter + vTrans
    where the rotation is applied around the origin (vCenter has already been
    absorbed into vTrans by ``_find_consensus_parameters``).

    NOTE: ``skimage.transform.rotate`` is designed for 2-D image arrays and
    operates on pixel grids — it is NOT appropriate for rotating (x, y) coordinate
    pairs.  A standard 2-D rotation matrix is used instead.

    :param cells: a list of cells
    :param consensus_translation: (tx, ty) translation in meters after rotation
    :param consensus_rotation_rad: rotation angle in radians
    :returns: predicted_positions list of (x, y) in meters
    """
    cos_a = np.cos(consensus_rotation_rad)
    sin_a = np.sin(consensus_rotation_rad)

    # 2-D rotation matrix  [[ cos, -sin], [sin,  cos]]
    # MATLAB convention: atan2(-mT(1,2), mT(1,1)) with mT = V*U'
    # mPosPred2 = mPos1 * mT'  (row-vector convention) + vTrans
    # mT' for angle θ is [[cos, sin], [-sin, cos]]  → x' = x*cos + y*sin  (transpose of standard)
    rotation_matrix_T = np.array([[cos_a, sin_a], [-sin_a, cos_a]])  # mT transposed

    tx, ty = consensus_translation
    predicted_positions = []
    for cell in cells:
        p = np.array(cell.center_reference)
        p_rot = rotation_matrix_T @ p          # equivalent to (mPos1 * mT') row-vector
        predicted_positions.append(
            (float(p_rot[0]) + tx, float(p_rot[1]) + ty)
        )
    return predicted_positions


def _get_distances_meters(
    cells: list[Cell],
    predicted_positions: list[tuple[float, float]],
) -> list[float]:
    """Calculate Euclidean distances of cells' comparison centers to predicted positions.

    :param cells: a list of cells
    :param predicted_positions: a list of predicted (x, y) positions in meters
    :returns: list of Euclidean distances in meters
    """
    distances = [
        float(np.sqrt(
            (cell.center_comparison[0] - pred[0]) ** 2
            + (cell.center_comparison[1] - pred[1]) ** 2
        ))
        for cell, pred in zip(cells, predicted_positions)
    ]
    return distances
