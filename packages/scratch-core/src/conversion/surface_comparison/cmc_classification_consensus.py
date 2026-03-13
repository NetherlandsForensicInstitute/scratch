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
        # All cells are CMC — skip pair search
        best_inliers_idx = list(range(n_filtered_cells))
    else:
        for idx_1 in range(n_filtered_cells):
            for idx_2 in range(idx_1 + 1, n_filtered_cells):

                # --- Seed: evaluate two-cell pair solution ---
                cell_distances, cell_angle_distances = _get_cmc_consensus(
                    [idx_1, idx_2], filtered_cells, reference_center
                )
                inliers_idx_current = np.where(
                    (cell_distances <= max_distance)
                    & (cell_angle_distances <= max_abs_angle_distance)
                )[0].tolist()

                criterion_current = _calculate_criterion(
                    inliers_idx_current, cell_distances, cell_angle_distances,
                    max_distance, max_abs_angle_distance,
                )

                if 2 < len(inliers_idx_current) < n_filtered_cells:
                    # --- Refinement: iteratively re-fit using all inliers ---
                    # make while loop pass for seed
                    inliers_idx_candidate = inliers_idx_current

                    while len(inliers_idx_candidate) == len(inliers_idx_current):
                        cell_distances, cell_angle_distances = _get_cmc_consensus(
                            inliers_idx_current, filtered_cells, reference_center
                        )
                        inliers_idx_candidate = np.where(
                            (cell_distances <= max_distance)
                            & (cell_angle_distances <= max_abs_angle_distance)
                        )[0].tolist()

                        criterion_candidate = _calculate_criterion(
                            inliers_idx_candidate, cell_distances, cell_angle_distances,
                            max_distance, max_abs_angle_distance,
                        )

                        # Accept if strictly more inliers, or same count with lower criterion
                        if len(inliers_idx_candidate) > len(inliers_idx_current) or (
                            len(inliers_idx_candidate) == len(inliers_idx_current)
                            and criterion_candidate < criterion_current
                        ):
                            criterion_current = criterion_candidate
                            inliers_idx_current = inliers_idx_candidate
                        else:
                            # make while loop break, also for len(inliers_idx_candidate) == len(inliers_idx_current)
                            break

                # --- Accept global best if this pair solution is better ---
                if len(inliers_idx_current) > len(best_inliers_idx) or (
                    len(inliers_idx_current) == len(best_inliers_idx)
                    and criterion_current < criterion
                ):
                    best_inliers_idx = inliers_idx_current
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
        consensus_translation, consensus_rotation_rad = _find_consensus_parameters(
            cmc_cells, reference_center
        )
        consensus_rotation_deg = float(np.degrees(consensus_rotation_rad))
    else:
        # then pick rotation and translation of the single is_congruent cell
        cell = [cell for cell in cells if cell.is_congruent == True][0]
        consensus_rotation_deg = cell.angle_deg
        consensus_translation = rotate(cell.center_reference) - cell.center_comparison

    return ComparisonResult(
        cells=updated_cells,
        shared_rotation=consensus_rotation_deg,
        shared_translation=consensus_translation,
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
    """Least-squares 'Procrustes' translation and rotation fit find consensus parameters.
    Explanation of the method:
    Say we have two coordinate-pair lists [X] and [Y] where X_i is coupled with Y_i. And we want to find the rotation for which:
    ||(X - rotation_center_X) R - (Y - rotation_center_Y)||F_2 is minimal. i.e. the Frobenius norm (the sum of squared distances between the transformed sets of points and a target set of points) is minimal.
    Now, we already know the rotation center for the reference grid X is and for the comparison grid Y: it is the center of the image, what we used to find the individual cell's rotations and centers. Normally, in Procrustes rotation we calculate the centers based on means of the data, but since in the end we want to compare the consensus rotation angle to the cell's rotation angle i think it's better to use the known rotation centers (contrary to what Matlab does). The rotation center for the comparison_grid Y is the mean of the centers_comparison of the cells . The optimal rotation can be found by completing the square in the Frobenius norm and observing that only the linear term
    -2trace(R^T X_centered^T Y_centered) depends on R. Therefore, this term should be minimal. Now, regard X_centered^T Y_centered = M with singular_value_decomposition(M) = U Sigma V^T, with U and V orthonormal basis and SIgma a diagonal eigenvalue matrix.
    For trace(R^T U Sigma V^T) to be maximal, since R, U and V are orthonormal matrices, you want: trace(R^T U Sigma V^T) = trace(Sigma). In order to achieve this (using the cyclic property of trace):
    trace(R^T U Sigma V^T) = trace(V^T R^T U Sigma), so R^T = V U^T, so R = U V^T.
    One last thing: since R is the collection of rotations and reflections, and physically we do not want reflections, we constrain the solution to reflections only be solving the above eigenvalue problem and, in case of reflection (determinant(R) = -1), reflecting the last axis (with the smallest eigenvalue, therefore yielding the minimal Frobenius norm given this contraint).

    :param cells: cells whose (center_reference, center_comparison) pairs are used for fitting
    :param reference_center: global center of rotation (x, y) in meters, used for rotation center of reference and comparison
    :returns: consensus_translation = 'comparison' - 'transformed_reference' (x, y) in meters, consensus_rotation_rad (float) in radians
    """
    if len(cells) == 0:
        raise ValueError("No cells found")

    centers_reference = np.array([cell.center_reference for cell in cells], dtype=float)   # (N, 2)
    centers_comparison = np.array([cell.center_comparison for cell in cells], dtype=float)  # (N, 2)
    center_of_rotation_reference = np.array(reference_center, dtype=float)                  # (2,)
    center_of_rotation_comparison = center_of_rotation_reference # we assume the same image_size and center.

    # Remove non_valid cells
    valid = ~(np.any(np.isnan(centers_reference), axis=1) | np.any(np.isnan(centers_comparison), axis=1))
    centers_reference = centers_reference[valid]
    centers_comparison = centers_comparison[valid]

    # Compute centroids
    mean_cell_centers_reference = centers_reference.mean(axis=0)   # mean of reference positions
    mean_cell_centers_comparison = centers_comparison.mean(axis=0)   # mean of comparison positions

    centers_reference_centered = centers_reference - center_of_rotation_reference       # centred reference positions
    centers_comparison_centered = centers_comparison - center_of_rotation_comparison       # centred comparison positions

    # SVD for best-fit rotation (no reflection)
    # M = centers_reference_centered^T * centers_comparison_centered
    M = centers_reference_centered.T @ centers_comparison_centered       # (2, 2)
    U, _, Vt = np.linalg.svd(M)
    rotation_matrix = U @ Vt  # (2, 2)

    # If det == -1 we have an unintended reflection; correct it by changing sign of lst column of U.
    if np.linalg.det(rotation_matrix) == -1:
        U[:, -1] *= -1
        rotation_matrix = U @ Vt

    # 'comparison' = 'rotated_reference' + translation, -> translation = 'comparison' - 'rotated_reference'.
    consensus_translation = (
        mean_cell_centers_comparison - center_of_rotation_comparison
        - (mean_cell_centers_reference - center_of_rotation_reference) @ rotation_matrix
    )

    # --- Rotation angle: atan2(-T[0,1], T[0,0])  (matches MATLAB atan2(-mT(1,2), mT(1,1))) ---
    consensus_rotation_rad = float(np.arctan2(-rotation_matrix[0, 1], rotation_matrix[0, 0]))
    consensus_translation = (float(consensus_translation[0]), float(consensus_translation[1]))

    return consensus_translation, consensus_rotation_rad

def _rotate_using_rotation_matrix(data: np.array, center: np.array, rotation_matrix: np.array) -> np.array:
    rotated = (data - center) @ rotation_matrix

    return rotated


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
    rotation_center: tuple[float, float]
) -> np.ndarray:
    """Predict reference_positions of cells in comparison frame after rotation and translation by consensus values.

    :param cells: a list of cells
    :param consensus_translation: (tx, ty) translation in meters after rotation
    :param consensus_rotation_rad: rotation angle in radians
    :param rotation_center: center of rotation, it is assumed the same for reference and comparison frame
    :returns: predicted_positions list of (x, y) in meters
    """
    cos_a = np.cos(consensus_rotation_rad)
    sin_a = np.sin(consensus_rotation_rad)

    # 2-D rotation matrix  [[ cos, -sin], [sin,  cos]]
    # MATLAB convention: atan2(-mT(1,2), mT(1,1)) with mT = V*U'
    # coord_pred = coord_pred * R  (row-vector convention) + vTrans
    # R for angle θ is [[cos, sin], [-sin, cos]]  → x' = x*cos + y*-sin
    rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    rotation_center = np.array(rotation_center).reshape(1,2)
    consensus_translation = np.array(consensus_translation).reshape(1,2)
    cell_centers_reference = np.array([cell.center_reference for cell in cells]) # (n, 2)
    reference_rotated = _rotate_using_rotation_matrix(cell_centers_reference, rotation_center, rotation_matrix)

    # predicted position in comparison frame = translation + rotation_center + rotation
    predicted_positions = consensus_translation + rotation_center + reference_rotated

    return predicted_positions


def _get_distances_meters(
    cells: list[Cell],
    predicted_positions: np.ndarray,
) -> list[float]:
    """Calculate Euclidean distances of cells' comparison centers to predicted positions.

    :param cells: a list of cells
    :param predicted_positions: a lnp.array predicted (x, y) positions in meters, shape (n, 2)
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
