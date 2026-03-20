import numpy as np

from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)


def classify_congruent_cells_consensus(
    cells: list[Cell], params: ComparisonParams, rotation_center_reference: list[float]
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
    :param rotation_center_reference: rotation center of reference image (meters). Used to predict coordinate when there is only one congruent cell.
    :returns: A `ComparisonResult` containing the classified cells, consensus
        rotation in degrees, and consensus translation in meters.
    :raises ValueError: If ``cells`` is empty.
    """
    if not cells:
        raise ValueError("Cannot identify CMC from an empty list.")

    minimum_fill_fraction = params.minimum_fill_fraction
    filtered_cells = _filter_cells(cells, minimum_fill_fraction)

    if not filtered_cells:
        raise ValueError(
            f"Cannot identify CMC. There are no cells with fill fraction >= {minimum_fill_fraction}."
        )

    n_filtered_cells = len(filtered_cells)

    max_distance = params.position_threshold  # in meters
    max_abs_angle_distance = params.angle_deviation_threshold  # in degrees

    # initialize solution: default to first cell as sole CMC
    best_inliers_idx = []
    criterion = np.inf

    if n_filtered_cells == 1 or (
        np.isinf(max_distance) and max_abs_angle_distance == 180
    ):
        # Then all filtered cells are inliers
        best_inliers_idx = list(range(n_filtered_cells))

    else:
        for idx_1 in range(n_filtered_cells):
            for idx_2 in range(idx_1 + 1, n_filtered_cells):
                # Seed: evaluate two-cell pair solution ---
                cell_distances, cell_angle_distances = _get_cmc_consensus(
                    [idx_1, idx_2], filtered_cells
                )
                inliers_idx_current = np.where(
                    (cell_distances <= max_distance)
                    & (cell_angle_distances <= max_abs_angle_distance)
                )[0].tolist()

                criterion_current = _calculate_criterion(
                    inliers_idx_current,
                    cell_distances,
                    cell_angle_distances,
                    max_distance,
                    max_abs_angle_distance,
                )

                if 2 < len(inliers_idx_current) < n_filtered_cells:
                    # Refinement: iteratively re-fit using all inliers ---
                    # make while loop pass for seed
                    inliers_idx_candidate = inliers_idx_current

                    while len(inliers_idx_candidate) == len(inliers_idx_current):
                        cell_distances, cell_angle_distances = _get_cmc_consensus(
                            inliers_idx_current, filtered_cells
                        )
                        inliers_idx_candidate = np.where(
                            (cell_distances <= max_distance)
                            & (cell_angle_distances <= max_abs_angle_distance)
                        )[0].tolist()

                        criterion_candidate = _calculate_criterion(
                            inliers_idx_candidate,
                            cell_distances,
                            cell_angle_distances,
                            max_distance,
                            max_abs_angle_distance,
                        )

                        # Accept if strictly more inliers, or same count with lower criterion
                        if len(inliers_idx_candidate) > len(inliers_idx_current) or (
                            len(inliers_idx_candidate) == len(inliers_idx_current)
                            and criterion_candidate < criterion_current
                        ):
                            criterion_current = criterion_candidate
                            inliers_idx_current = inliers_idx_candidate
                        else:
                            # break while loop, also for len(inliers_idx_candidate) == len(inliers_idx_current) and criterion did not improve
                            break

                # --- Accept global best if the current solution is better ---
                if len(inliers_idx_current) > len(best_inliers_idx) or (
                    len(inliers_idx_current) == len(best_inliers_idx)
                    and criterion_current < criterion
                ):
                    best_inliers_idx = inliers_idx_current
                    criterion = criterion_current

            if len(best_inliers_idx) == n_filtered_cells:
                break  # outer loop short-circuit

    # --- Mark cells as congruent ---
    congruent_set = set(filtered_cells[i] for i in best_inliers_idx)

    # Update is_congruent labels based on congruent_set
    for cell in cells:
        cell.is_congruent = cell in congruent_set

    # --- Estimate shared rotation and transformation
    if len(best_inliers_idx) > 1:
        # From CMC inliers
        cmc_cells = [filtered_cells[i] for i in best_inliers_idx]
        consensus_translation, consensus_rotation_rad, _, _ = (
            _find_consensus_parameters(cmc_cells)
        )
        consensus_rotation_deg = float(np.degrees(consensus_rotation_rad))
        consensus_translation = (consensus_translation[0], consensus_translation[1])
    else:
        # There was only one filtered_cell
        congruent_cell = filtered_cells[0]
        predicted_coordinate = list(
            _rotate_using_angle_deg(
                np.array(congruent_cell.center_reference),
                -congruent_cell.angle_deg,
                np.array(rotation_center_reference),
            )[0]
            + np.array(rotation_center_reference)
        )
        consensus_translation = tuple(
            [
                c - r
                for c, r in zip(congruent_cell.center_comparison, predicted_coordinate)
            ]
        )
        consensus_rotation_deg = congruent_cell.angle_deg

    return ComparisonResult(
        cells=cells,
        shared_rotation=consensus_rotation_deg,
        shared_translation=consensus_translation,
    )


def _filter_cells(cells: list[Cell], minimum_fill_fraction: float) -> list[Cell]:
    """Keep cells that have fill_fraction_reference >= minimum_fill_fraction.

    :param cells: a list of Cells to filter
    :param minimum_fill_fraction: minimum fill fraction of filtered cells
    :returns: filtered_cells, a list of filtered cells
    """
    filtered_cells = [
        cell for cell in cells if cell.fill_fraction_reference >= minimum_fill_fraction
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate cell_distances and cell_angle_distances for all cells after finding consensus parameters.

    :param included_idx: a list of included cell indices (used for least-squares fit)
    :param cells: a list of all filtered cells
    :param reference_rotation_center: global center [x, y] used as the fixed rotation point (meters)
    :returns: cell_distances (meters), cell_angle_distances in absolute degrees — both as np.ndarray of length len(cells)
    """
    cells_for_least_squares = [cells[idx] for idx in included_idx]
    (
        _,
        consensus_rotation_rad,
        reference_rotation_center,
        comparison_rotation_center,
    ) = _find_consensus_parameters(cells_for_least_squares)
    cell_distances, cell_angle_distances = _get_distances(
        cells,
        consensus_rotation_rad,
        rotation_center_reference=reference_rotation_center,
        rotation_center_comparison=comparison_rotation_center,
    )
    return cell_distances, cell_angle_distances


def _find_consensus_parameters(
    cells: list[Cell],
) -> tuple[tuple[float, float], float, tuple[float, float], tuple[float, float]]:
    """Least-squares 'Procrustes' rotation fit to find consensus rotation and translation parameters.

    Explanation of the method:
    Say we have two coordinate-pair lists [X] and [Y] where X_i is coupled with Y_i. And we want to find the rotation matrix R and translation of X to Y for which:
    ||(X - rotation_center_X) R - (Y - rotation_center_Y)||F_2 is minimal. i.e. the Frobenius norm (the sum of squared distances between a linearly transformed set of points and a target set of points) is minimal.
    The rotation_centra yielding minimum Frobenius norm for the rotation operation are the coordinate means of X and Y. Note that this is a different coordinate system than the one used to find the rotation of individual cells during registration but this does not matter since optimal rotation angle is independent of coordinate system. We just want to find this rotation by minimizing the Frobenius norm and this minimizes the norm.
    It immediately follows that optimal translation in this coordinate system is rotation_center_Y - rotation_center_X.
    The optimal rotation can be found by completing the square in the Frobenius norm and observing that only the linear term
    -2trace(R^T X_centered^T Y_centered) depends on R. Therefore, this term should be minimal. Now, regard X_centered^T Y_centered = M with singular_value_decomposition(M) = U Sigma V^T, with U and V orthonormal basis and SIgma a diagonal eigenvalue matrix.
    For trace(R^T U Sigma V^T) to be maximal, since R, U and V are orthonormal matrices, you want: trace(R^T U Sigma V^T) = trace(Sigma). In order to achieve this (using the cyclic property of trace):
    trace(R^T U Sigma V^T) = trace(V^T R^T U Sigma), so R^T = V U^T, so R = U V^T.
    One last thing: since R is the collection of rotations and reflections, and physically we do not want reflections, we constrain the solution to reflections only be solving the above eigenvalue problem and, in case of reflection (determinant(R) = -1), reflecting the last axis (with the smallest eigenvalue, therefore yielding the minimal Frobenius norm given this contraint).

    :param cells: cells whose (center_reference, center_comparison) pairs are used for fitting
    :returns: consensus_translation, consensus_rotation_rad of reference (float) in radians, rotation_center_reference, rotation_center_comparison
    """
    if len(cells) == 0:
        raise ValueError("No cells found")

    centers_reference = np.array(
        [cell.center_reference for cell in cells], dtype=float
    )  # (N, 2)
    centers_comparison = np.array(
        [cell.center_comparison for cell in cells], dtype=float
    )  # (N, 2)

    # Remove non_valid cells
    valid = ~(
        np.any(np.isnan(centers_reference), axis=1)
        | np.any(np.isnan(centers_comparison), axis=1)
    )
    centers_reference = centers_reference[valid]
    centers_comparison = centers_comparison[valid]

    # Compute centroids
    rotation_center_reference = centers_reference.mean(
        axis=0
    )  # mean of reference positions (2,)
    rotation_center_comparison = centers_comparison.mean(
        axis=0
    )  # mean of comparison positions (2,)

    centers_reference_centered = (
        centers_reference - rotation_center_reference
    )  # centred reference positions
    centers_comparison_centered = (
        centers_comparison - rotation_center_comparison
    )  # centred comparison positions

    # SVD for best-fit rotation (no reflection)
    # M = centers_reference_centered^T * centers_comparison_centered
    M = centers_reference_centered.T @ centers_comparison_centered  # (2, 2)
    U, _, Vt = np.linalg.svd(M)
    rotation_matrix = U @ Vt  # (2, 2)

    # If det == -1 we have an unintended reflection; correct it by changing sign of last column of U.
    if np.linalg.det(rotation_matrix) == -1:
        U[:, -1] *= -1
        rotation_matrix = U @ Vt

    # Rotation angle: atan2(sin/cos)
    (cos, sin) = tuple(rotation_matrix[0])
    consensus_rotation_rad = float(np.arctan2(sin, cos))

    consensus_translation = rotation_center_comparison - rotation_center_reference

    return (
        consensus_translation,
        consensus_rotation_rad,
        rotation_center_reference,
        rotation_center_comparison,
    )


def _rotation_component_with_rotation_matrix(
    data: np.ndarray, center: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate data around center, return only rotation component (no offset by center).

    :param data: data to be rotated, shape (n ,m), n cases with m features
    :param center: center of rotation, shape (1 ,m)
    :param rotation_matrix: rotation matrix, shape (m, m)

    :returns rotated data minus center, shape (n ,m)
    """

    rotated = (data - center) @ rotation_matrix

    return rotated


def _get_distances(
    cells: list[Cell],
    consensus_rotation_rad: float,
    rotation_center_reference: tuple[float, float],
    rotation_center_comparison: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Get distances and abs_angle_distances of cell locations/rotations versus
    consensus_translation and consensus_rotation_rad.

    :param cells: a list of cells
    :param consensus_rotation_rad: rotation in radians (used to predict comparison positions
        AND to compute angle residuals after converting to degrees)
    :param rotation_center_reference: center of rotation in reference frame, shape (1 ,m)
    :param rotation_center_comparison: center of rotation in comparison frame, shape (1 ,m)
    :returns: distances (meters) as np.ndarray, abs_angle_distances (unsigned degrees) as np.ndarray
    """
    predicted_positions = _predict_positions(
        cells,
        consensus_rotation_rad,
        rotation_center_reference,
        rotation_center_comparison,
    )
    distances = np.array(_get_distances_meters(cells, predicted_positions))

    consensus_rotation_deg = float(np.degrees(consensus_rotation_rad))

    # The absolute residual is |cell.angle_deg - -consensus_rotation_deg|, since we use pixel_coordinates for rotation_angle of cells and mathematical coordinates here.
    abs_angle_distances = np.array(
        [abs(cell.angle_deg - -consensus_rotation_deg) for cell in cells]
    )

    return distances, abs_angle_distances


def _rotate_using_angle_deg(
    xy_data: np.ndarray, angle_deg: float, reference_center: np.ndarray
) -> np.ndarray:
    """Rotate data around center.

    :param xy_data: data to be rotated, shape (n ,m), n cases with m features
    :param angle_deg: angle in degrees
    :param reference_center: center of rotation, shape (m)
    :returns rotated data, shape (n ,m)
    """
    reference_center = reference_center.reshape(1, -1)
    angle_rad = np.radians(angle_deg)
    rotation_matrix = _build_2d_rotation_matrix(angle_rad)

    return _rotation_component_with_rotation_matrix(
        data=xy_data, center=reference_center, rotation_matrix=rotation_matrix
    )


def _build_2d_rotation_matrix(angle_rad: float) -> np.ndarray:
    """Build 2d rotation matrix from angle_rad.

     2-D rotation matrix  [[ cos, -sin], [sin,  cos]]
     R for angle θ is [row1, row2] = [[cos, sin], [-sin, cos]]  → x' = x*cos + y*-sin

    :param angle_rad: angle in radians
    :returns: 2d rotation matrix, shape (2,2).
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    return rotation_matrix


def _predict_positions(
    cells: list[Cell],
    consensus_rotation_rad: float,
    rotation_center_reference: tuple[float, float],
    rotation_center_comparison: tuple[float, float],
) -> np.ndarray:
    """Predict reference_positions of cells in comparison frame after rotation and translation by consensus values.

    :param cells: a list of cells
    :param consensus_rotation_rad: rotation angle in radians
    :param rotation_center_reference:
    :param rotation_center_comparison:
    :returns: predicted_positions list of (x, y) in meters
    """

    rotation_matrix = _build_2d_rotation_matrix(consensus_rotation_rad)
    rotation_center = np.array(rotation_center_reference).reshape(1, 2)
    cell_centers_reference = np.array(
        [cell.center_reference for cell in cells]
    )  # (n, 2)
    references_rotated = _rotation_component_with_rotation_matrix(
        data=cell_centers_reference,
        center=rotation_center,
        rotation_matrix=rotation_matrix,
    )

    # predicted position in comparison frame = rotation + center
    predicted_positions = references_rotated + rotation_center_comparison

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
        float(
            np.sqrt(
                (cell.center_comparison[0] - pred[0]) ** 2
                + (cell.center_comparison[1] - pred[1]) ** 2
            )
        )
        for cell, pred in zip(cells, predicted_positions)
    ]
    return distances
