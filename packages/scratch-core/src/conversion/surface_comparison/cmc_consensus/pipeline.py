import numpy as np
from itertools import combinations

from conversion.surface_comparison.cmc_consensus.criterion import (
    _get_cell_angle_and_position_distances,
    calculate_criterion,
)
from conversion.surface_comparison.cmc_consensus.procrustes import (
    find_consensus_parameters,
    _get_rotation_component_using_angle_degree,
)
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
    ComparisonParams,
)

from conversion.surface_comparison.cmc_consensus.models import (
    CMCTranslationRotation,
)


def classify_congruent_cells_consensus(
    cells: list[Cell], params: ComparisonParams, reference_center: tuple[float, float]
) -> ComparisonResult:
    """
    Identify Congruent Matching Cells (CMCs) using a consensus-based procedure using Procrustes rotation and translation
    to find consensus parameters

    Steps:
    1. Filter cells that pass the similarity threshold.
    2. Loop over all pairs (i,j) of those cells, and for each pair:
       Estimate a rigid body transformation (rotation + translation) from just those two cells via _get_cell_angle_and_position_distances → _find_consensus_parameters
       Find all other cells that fall within position_threshold and angle_deviation_threshold of that predicted location.
       Attempt to iteratively refine by re-fitting using all successful cells.
       Keep the solution if it yields more CMC cells than the current best (or equal count with better quality).
    3. Get a boolean vector flagging which cells are CMC.
    4. Return a ComparisonResult.

    :param cells: Per-cell registration results to classify.
    :param params: Algorithm parameters (thresholds for score, angle, and position).
    :param reference_center: rotation center of reference image (meters). Used to predict coordinate when there is only one congruent cell.
    :returns: A `ComparisonResult` containing the classified cells, consensus
        rotation in degrees, and consensus translation in meters.
    :raises ValueError: If ``cells`` is empty.
    """

    if len(cells) == 1:
        # Then this cell is an inlier by definition
        best_ids = [0]

    else:
        best_ids = _find_best_ids(
            cells, params.position_threshold, params.angle_deviation_threshold
        )

    _update_congruent_cells(cells, best_ids)

    consensus = _get_estimated_translation_rotation(cells, reference_center)

    return ComparisonResult(
        cells=cells,
        estimated_rotation=consensus.rotation,
        estimated_translation=consensus.translation,
    )


def _find_best_ids(
    cells: list[Cell], max_distance: float, max_abs_angle_distance: float
) -> list[int]:
    """Core algorithm to find the best inlier ids. Loop over all indices pairs as initial solution, and iteratively refine this solution. Update global solution if refinement has more cells or if criterion improves for same amount of cells.

    :param cells: list of cells.
    :param max_distance: maximum distance to consider for consensus, in meters.
    :param max_abs_angle_distance: maximum absolute angle deviation to consider for consensus, in degrees.

    :returns: list of inlier cell ids, these will be the congruent cells
    """

    best_ids = []
    criterion = np.inf
    n_cells = len(cells)

    for pair_ids in combinations(range(n_cells), 2):
        # Initial solution: evaluate two-cell pair solution ---
        cell_distances, cell_angle_distances = _get_cell_angle_and_position_distances(
            list(pair_ids), cells
        )
        current_ids = np.where(
            (cell_distances <= max_distance)
            & (cell_angle_distances <= max_abs_angle_distance)
        )[0].tolist()

        criterion_current = calculate_criterion(
            cell_distances[current_ids],
            cell_angle_distances[current_ids],
            max_distance,
            max_abs_angle_distance,
        )

        if 2 < len(current_ids) < n_cells:
            _refine(
                current_ids,
                criterion_current,
                cells,
                max_distance,
                max_abs_angle_distance,
            )

        # Accept current solution if it is better
        if len(current_ids) > len(best_ids) or (
            len(current_ids) == len(best_ids) and criterion_current < criterion
        ):
            best_ids = current_ids
            criterion = criterion_current

        if len(best_ids) == n_cells:
            return best_ids  # outer loop short-circuit

    return best_ids


def _update_congruent_cells(cells: list[Cell], congruent_ids: list[int]) -> None:
    """update cell.is_congruent property
    :param cells: list of cells.
    :param congruent_ids: list of cell ids that are congruent
    """

    for i, cell in enumerate(cells):
        cell.is_congruent = i in set(congruent_ids)


def _get_estimated_translation_rotation(
    cells: list[Cell], reference_center: tuple[float, float]
) -> CMCTranslationRotation:
    """Calculate shared rotation and transformation
    :param cells: list of cells.
    :param reference_center: reference center
    :returns: shared rotation and transformation, in CMCTranslationRotation
    """
    cmc_cells = [cell for cell in cells if cell.is_congruent]

    if len(cmc_cells) > 1:
        consensus_parameters = find_consensus_parameters(cmc_cells)
        consensus_rotation_deg = float(np.degrees(consensus_parameters.rotation_rad))
        consensus_translation = (
            consensus_parameters.translation[0],
            consensus_parameters.translation[1],
        )
    else:
        # There was only one congruent cell
        congruent_cell = cells[0]
        predicted_coordinate = list(
            _get_rotation_component_using_angle_degree(
                np.array(congruent_cell.center_reference),
                -congruent_cell.angle_deg,
                np.array(reference_center),
            )[0]
            + np.array(reference_center)
        )
        consensus_translation = tuple(
            [
                center_float - reference_float
                for center_float, reference_float in zip(
                    congruent_cell.center_comparison, predicted_coordinate
                )
            ]
        )
        consensus_rotation_deg = congruent_cell.angle_deg

    shared_parameters = CMCTranslationRotation(
        translation=consensus_translation, rotation=consensus_rotation_deg
    )

    return shared_parameters


def _refine(
    current_ids: list[int],
    criterion_current: float,
    cells: list[Cell],
    max_distance: float,
    max_abs_angle_distance: float,
) -> None:
    """iteratively re-fit current_ids and criterion_current

    :param current_ids: a list of inlier indices (used for least-squares Procrustus fit)
    :param criterion_current: the current value of the criterion
    :param cells: a list of cells
    :param max_distance: maximum distance threshold (meters)
    :param max_abs_angle_distance: maximum absolute angle threshold (degrees)
    """

    while True:
        cell_distances, cell_angle_distances = _get_cell_angle_and_position_distances(
            current_ids, cells
        )
        candidate_ids = np.where(
            (cell_distances <= max_distance)
            & (cell_angle_distances <= max_abs_angle_distance)
        )[0].tolist()

        criterion_candidate = calculate_criterion(
            cell_distances[candidate_ids],
            cell_angle_distances[candidate_ids],
            max_distance,
            max_abs_angle_distance,
        )

        # Accept if strictly more inlier, or same count with lower criterion
        if len(candidate_ids) > len(current_ids) or (
            len(candidate_ids) == len(current_ids)
            and criterion_candidate < criterion_current
        ):
            criterion_current = criterion_candidate
            current_ids = candidate_ids
        else:
            # we have our local optimum and return, also for len(candidate_ids) == len(current_ids) and criterion did not improve
            return
