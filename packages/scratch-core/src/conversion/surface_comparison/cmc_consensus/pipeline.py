import numpy as np

from conversion.surface_comparison.cmc_consensus.criterion import calculate_criterion
from conversion.surface_comparison.cmc_consensus.models import (
    CMCTranslationRotation,
)
from conversion.surface_comparison.cmc_consensus.procrustes import (
    find_consensus_parameters,
    get_rotation_angle_closed_form,
    get_rotation_component_using_angle_degree,
    get_translation_about,
)
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
)
from conversion.surface_comparison.utils import rotate_points, wrap_angles

_PAIR_CHUNK = 1024  # number of seed pairs evaluated per vectorized batch
FitCache = dict[tuple[int, ...], tuple[list[int], float]]


def classify_congruent_cells_consensus(
    cells: list[Cell],
    angle_deviation_threshold: float,
    correlation_threshold: float,
    position_threshold: float,
    reference_center: tuple[float, float],
) -> ComparisonResult:
    """
    Identify Congruent Matching Cells (CMCs) with a consensus-based procedure using Procrustes rotation and translation
    to find consensus parameters

    Steps:
    1. Iteratively refine rigid body transformation (Procrustes rotation with scale fixed) started from cell pairs,
        keeping solutions with more cells or better quality.
    2. Flag the geometric inliers that also pass the similarity threshold as CMC.
    3. Estimate the consensus rotation and translation from the geometric inliers.
    4. Return a ComparisonResult.

    :param cells: Per-cell registration results to classify.
    :param angle_deviation_threshold: Maximum angle deviation from consensus.
    :param correlation_threshold: Minimum correlation to classify as congruent.
    :param position_threshold: Maximum positional deviation from consensus.
    :param reference_center: Reference rotation center (meters); used if only one cell fits.
    :returns: A `ComparisonResult` containing the classified cells, consensus rotation in degrees, and consensus
        translation in meters, both expressed around `reference_center`. Both are NaN when no consensus
        geometry is found; every cell is then non-congruent.
    """
    if len(cells) == 1:
        # Then this cell is an inlier by definition
        inlier_ids = [0]
    else:
        inlier_ids = _find_best_ids(
            cells=cells,
            max_distance=position_threshold,
            max_abs_angle_distance=angle_deviation_threshold,
        )
    if not inlier_ids:
        # Note: the `meta_data` residuals keep their pre-classification values here.
        # Only `is_congruent` (False for every cell here) is acted on downstream.
        return ComparisonResult(
            cells=cells,
            estimated_rotation=float("nan"),
            estimated_translation=(float("nan"), float("nan")),
        )
    consensus = _get_estimated_translation_rotation(
        cells=[cells[i] for i in inlier_ids], reference_center=reference_center
    )
    # Apply the similarity threshold to the inliers
    cmc_ids = [i for i in inlier_ids if cells[i].best_score >= correlation_threshold]
    _update_congruent_cells(cells=cells, congruent_ids=cmc_ids)
    _update_cell_meta_data(
        cells=cells,
        inlier_ids=inlier_ids,
        consensus=consensus,
        reference_center=reference_center,
    )

    return ComparisonResult(
        cells=cells,
        estimated_rotation=consensus.rotation,
        estimated_translation=consensus.translation,
    )


def _find_best_ids(
    cells: list[Cell], max_distance: float, max_abs_angle_distance: float
) -> list[int]:
    """
    Find best inliers by iteratively refining initial pair-based solutions, prioritizing higher cell count than better
    criterion. All pair fits are evaluated in vectorized batches; refinement fits are memoized.

    :param cells: list of cells.
    :param max_distance: maximum distance to consider for consensus, in meters.
    :param max_abs_angle_distance: maximum absolute angle deviation to consider for consensus, in degrees.
    :returns: list of inlier cell ids, these will be the congruent cells
    """
    n_cells = len(cells)
    reference_centers, comparison_centers, cell_angles_rad = _get_cell_arrays(cells)
    cache: FitCache = {}

    best_ids: list[int] = []
    best_criterion = np.inf
    first, second = np.triu_indices(
        n_cells, k=1
    )  # same order as itertools.combinations

    for start in range(0, len(first), _PAIR_CHUNK):
        i, j = first[start : start + _PAIR_CHUNK], second[start : start + _PAIR_CHUNK]
        # Two-point fit: centroids are the midpoints
        reference_centroids = (reference_centers[i] + reference_centers[j]) / 2
        comparison_centroids = (comparison_centers[i] + comparison_centers[j]) / 2
        reference_pairs = np.stack(
            [
                reference_centers[i] - reference_centroids,
                reference_centers[j] - reference_centroids,
            ],
            axis=1,
        )
        comparison_pairs = np.stack(
            [
                comparison_centers[i] - comparison_centroids,
                comparison_centers[j] - comparison_centroids,
            ],
            axis=1,
        )
        rotations = get_rotation_angle_closed_form(
            centered_reference_centers=reference_pairs,
            centered_comparison_centers=comparison_pairs,
        )

        distances, angle_distances = _get_residuals(
            reference_centers=reference_centers,
            comparison_centers=comparison_centers,
            cell_angles_rad=cell_angles_rad,
            rotations=rotations,
            reference_centroids=reference_centroids,
            comparison_centroids=comparison_centroids,
        )
        inlier_mask = (distances <= max_distance) & (
            angle_distances <= max_abs_angle_distance
        )
        counts = inlier_mask.sum(axis=1)

        for row in range(len(i)):
            count = int(counts[row])
            # A seed with <= 2 inliers is not refined, so it can only win while the best has <= 2 cells
            if count <= 2 and count < len(best_ids):
                continue
            current_ids = np.flatnonzero(inlier_mask[row]).tolist()
            current_criterion = calculate_criterion(
                distances[row, current_ids],
                angle_distances[row, current_ids],
                max_distance,
                max_abs_angle_distance,
            )
            if 2 < count < n_cells:
                current_ids, current_criterion = _refine_inliers(
                    current_ids=current_ids,
                    current_criterion=current_criterion,
                    reference_centers=reference_centers,
                    comparison_centers=comparison_centers,
                    cell_angles_rad=cell_angles_rad,
                    max_distance=max_distance,
                    max_abs_angle_distance=max_abs_angle_distance,
                    cache=cache,
                )
            if len(current_ids) > len(best_ids) or (
                len(current_ids) == len(best_ids) and current_criterion < best_criterion
            ):
                best_ids, best_criterion = current_ids, current_criterion
            if len(best_ids) == n_cells:
                return best_ids  # outer loop short-circuit

    return best_ids


def _update_congruent_cells(cells: list[Cell], congruent_ids: list[int]) -> None:
    """
    Update the cell.is_congruent property.

    :param cells: list of cells, updated in place.
    :param congruent_ids: list of cell ids that are congruent
    """

    congruent = set(congruent_ids)
    for i, cell in enumerate(cells):
        cell.is_congruent = i in congruent


def _update_cell_meta_data(
    cells: list[Cell],
    inlier_ids: list[int],
    consensus: CMCTranslationRotation,
    reference_center: tuple[float, float],
) -> None:
    """
    Record each cell's residuals against the consensus pose, the same fields the median classifier fills.

    :param cells: list of cells, updated in place.
    :param inlier_ids: ids of the cells the consensus pose was fitted on.
    :param consensus: consensus rotation (degrees) and translation (meters) about `reference_center`.
    :param reference_center: reference rotation center (meters)
    """
    inliers = set(inlier_ids)
    predicted_positions = (
        rotate_points(
            points=np.array([cell.center_reference for cell in cells]),
            angle=-np.radians(consensus.rotation),
            center=reference_center,
        )
        + consensus.translation
    )

    for i, (cell, predicted) in enumerate(zip(cells, predicted_positions)):
        cell.meta_data.is_outlier = i not in inliers
        cell.meta_data.residual_angle_deg = float(
            np.degrees(wrap_angles(np.radians(cell.angle_deg - consensus.rotation)))
        )
        cell.meta_data.position_error = (
            float(cell.center_comparison[0] - predicted[0]),
            float(cell.center_comparison[1] - predicted[1]),
        )


def _get_estimated_translation_rotation(
    cells: list[Cell], reference_center: tuple[float, float]
) -> CMCTranslationRotation:
    """
    Calculate the shared rotation and translation of the given cells.

    :param cells: list of (inlier) cells
    :param reference_center: reference rotation center (meters)
    :returns: shared rotation (degrees) and translation (meters) about `reference_center`
    """
    if len(cells) > 1:
        consensus_parameters = find_consensus_parameters(cells)
        # Negate to match the angle convention of Cell.angle_deg.
        consensus_rotation_deg = -float(np.degrees(consensus_parameters.rotation_rad))
        consensus_translation = get_translation_about(
            consensus_parameters=consensus_parameters, center=reference_center
        )
    else:
        # There is only one cell
        inlier_cell = cells[0]
        predicted_coordinate = list(
            get_rotation_component_using_angle_degree(
                xy_data=np.array(inlier_cell.center_reference),
                angle_deg=-inlier_cell.angle_deg,
                reference_center=np.array(reference_center),
            )[0]
            + np.array(reference_center)
        )
        consensus_translation = (
            float(inlier_cell.center_comparison[0] - predicted_coordinate[0]),
            float(inlier_cell.center_comparison[1] - predicted_coordinate[1]),
        )
        consensus_rotation_deg = inlier_cell.angle_deg

    return CMCTranslationRotation(
        translation=consensus_translation, rotation=consensus_rotation_deg
    )


def _get_residuals(
    reference_centers: np.ndarray,
    comparison_centers: np.ndarray,
    cell_angles_rad: np.ndarray,
    rotations: np.ndarray,
    reference_centroids: np.ndarray,
    comparison_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distance and absolute angle residuals of all n cells under P rigid fits at once.

    Each fit p rotates the reference centers by `rotations[p]` about `reference_centroids[p]` and moves that centroid
    onto `comparison_centroids[p]`.

    :param reference_centers: reference centers of all cells, shape (n, 2), in meters.
    :param comparison_centers: comparison centers of all cells, shape (n, 2), in meters.
    :param cell_angles_rad: cell angles, shape (n,), in radians.
    :param rotations: fitted rotation angles, shape (P,), in radians.
    :param reference_centroids: rotation centers in reference coordinates, shape (P, 2), in meters.
    :param comparison_centroids: matching centers in comparison coordinates, shape (P, 2), in meters.
    :returns: (distances in meters, absolute angle residuals in degrees), both of shape (P, n).
    """
    cos, sin = np.cos(rotations)[:, None], np.sin(rotations)[:, None]
    dx = reference_centers[None, :, 0] - reference_centroids[:, 0:1]
    dy = reference_centers[None, :, 1] - reference_centroids[:, 1:2]
    # Same convention as `(data - center) @ _build_2d_rotation_matrix(rotation)`
    predicted_x = dx * cos - dy * sin + comparison_centroids[:, 0:1]
    predicted_y = dx * sin + dy * cos + comparison_centroids[:, 1:2]
    distances = np.sqrt(
        (comparison_centers[None, :, 0] - predicted_x) ** 2
        + (comparison_centers[None, :, 1] - predicted_y) ** 2
    )
    # Cell angles are in pixel coordinates and the fit in math coordinates, hence the +
    angle_distances = np.abs(
        np.degrees(wrap_angles(cell_angles_rad[None, :] + rotations[:, None]))
    )
    return distances, angle_distances


def _get_cell_arrays(cells: list[Cell]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect the cell positions and angles into arrays for vectorized computation.

    :param cells: list of n cells.
    :returns: (reference centers (n, 2), comparison centers (n, 2), cell angles in radians (n,))
    """
    return (
        np.array([cell.center_reference for cell in cells], dtype=float),
        np.array([cell.center_comparison for cell in cells], dtype=float),
        np.radians([cell.angle_deg for cell in cells]),
    )


def _fit_procrustes(
    ids: list[int],
    reference_centers: np.ndarray,
    comparison_centers: np.ndarray,
    cell_angles_rad: np.ndarray,
    max_distance: float,
    max_abs_angle_distance: float,
    cache: FitCache,
) -> tuple[list[int], float]:
    """
    Procrustes fit on the given ids, returning the inliers among all cells and their criterion. Memoized in `cache`.

    :param ids: cell ids to fit on.
    :param reference_centers: reference centers of all cells, shape (n, 2), in meters.
    :param comparison_centers: comparison centers of all cells, shape (n, 2), in meters.
    :param cell_angles_rad: cell angles, shape (n,), in radians.
    :param max_distance: maximum distance threshold (meters)
    :param max_abs_angle_distance: maximum absolute angle threshold (degrees)
    :param cache: memo of previous fits, keyed by id tuple; updated in place.
    :returns: (inlier ids, criterion)
    """
    key = tuple(ids)
    if key in cache:
        return cache[key]
    reference_centroid = reference_centers[ids].mean(axis=0)
    comparison_centroid = comparison_centers[ids].mean(axis=0)
    rotation = get_rotation_angle_closed_form(
        centered_reference_centers=reference_centers[ids] - reference_centroid,
        centered_comparison_centers=comparison_centers[ids] - comparison_centroid,
    )
    distances, angle_distances = _get_residuals(
        reference_centers=reference_centers,
        comparison_centers=comparison_centers,
        cell_angles_rad=cell_angles_rad,
        rotations=np.atleast_1d(rotation),
        reference_centroids=reference_centroid[None],
        comparison_centroids=comparison_centroid[None],
    )
    distances, angle_distances = distances[0], angle_distances[0]
    inliers = np.flatnonzero(
        (distances <= max_distance) & (angle_distances <= max_abs_angle_distance)
    ).tolist()
    cache[key] = (
        inliers,
        calculate_criterion(
            distances[inliers],
            angle_distances[inliers],
            max_distance,
            max_abs_angle_distance,
        ),
    )
    return cache[key]


def _refine_inliers(
    current_ids: list[int],
    current_criterion: float,
    reference_centers: np.ndarray,
    comparison_centers: np.ndarray,
    cell_angles_rad: np.ndarray,
    max_distance: float,
    max_abs_angle_distance: float,
    cache: FitCache,
) -> tuple[list[int], float]:
    """
    Iteratively re-fit inlier set and criterion until a local optimum is reached.

    :param current_ids: a list of inlier indices (used for least-squares Procrustes fit)
    :param current_criterion: the current value of the criterion
    :param reference_centers: reference centers of all cells, shape (n, 2), in meters.
    :param comparison_centers: comparison centers of all cells, shape (n, 2), in meters.
    :param cell_angles_rad: cell angles, shape (n,), in radians.
    :param max_distance: maximum distance threshold (meters)
    :param max_abs_angle_distance: maximum absolute angle threshold (degrees)
    :param cache: memo of previous fits, shared across refinements; updated in place.
    :returns: tuple of (updated inlier indices, updated criterion)
    """
    best_ids, best_criterion = current_ids, current_criterion
    while True:
        candidate_ids, candidate_criterion = _fit_procrustes(
            ids=best_ids,
            reference_centers=reference_centers,
            comparison_centers=comparison_centers,
            cell_angles_rad=cell_angles_rad,
            max_distance=max_distance,
            max_abs_angle_distance=max_abs_angle_distance,
            cache=cache,
        )
        # Accept if strictly more inliers, or same count with lower criterion
        if len(candidate_ids) > len(best_ids) or (
            len(candidate_ids) == len(best_ids) and candidate_criterion < best_criterion
        ):
            best_ids, best_criterion = candidate_ids, candidate_criterion
        else:
            return best_ids, best_criterion
