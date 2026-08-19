from itertools import combinations

import numpy as np

from conversion.surface_comparison.cmc_consensus.criterion import (
    _get_cell_angle_and_position_distances,
    calculate_criterion,
)
from conversion.surface_comparison.cmc_consensus.models import (
    CMCTranslationRotation,
)
from conversion.surface_comparison.cmc_consensus.procrustes import (
    _get_rotation_component_using_angle_degree,
    find_consensus_parameters,
    get_translation_about,
)
from conversion.surface_comparison.models import (
    NO_CONSENSUS_ROTATION,
    NO_CONSENSUS_TRANSLATION,
    Cell,
    ComparisonParams,
    ComparisonResult,
)


def classify_congruent_cells_consensus(
    cells: list[Cell], params: ComparisonParams, reference_center: tuple[float, float]
) -> ComparisonResult:
    """
    Identify Congruent Matching Cells (CMCs) using a consensus-based procedure using Procrustes rotation and translation
    to find consensus parameters

    Steps:
    1. Iteratively refine rigid body transformations from cell pairs, keeping solutions with more cells or better quality.
    2. Flag the geometric inliers that also pass the similarity threshold as CMC.
    3. Estimate the consensus rotation and translation from the geometric inliers.
    4. Return a ComparisonResult.

    :param cells: Per-cell registration results to classify.
    :param params: Algorithm parameters (thresholds for score, angle, and position).
    :param reference_center: Reference rotation center (meters); used if only one cell fits.
    :returns: A `ComparisonResult` containing the classified cells, consensus rotation in degrees, and consensus
        translation in meters, both expressed around `reference_center`. When no consensus geometry is found the
        rotation is zero and the translation is NaN.
    :raises ValueError: If ``cells`` is empty.
    """
    if not cells:
        raise ValueError("Cannot identify CMC from an empty list.")

    if len(cells) == 1:
        # Then this cell is an inlier by definition
        inlier_ids = [0]
    else:
        inlier_ids = _find_best_ids(
            cells, params.position_threshold, params.angle_deviation_threshold
        )
    # Apply the similarity threshold to the inliers
    cmc_ids = [
        i for i in inlier_ids if cells[i].best_score >= params.correlation_threshold
    ]
    _update_congruent_cells(cells, cmc_ids)

    if not inlier_ids:
        # No consensus geometry was found, so there is no pose to report.
        # TODO: report NaN for the rotation too, once the API schema allows a nullable rotation.
        return ComparisonResult(
            cells=cells,
            estimated_rotation=NO_CONSENSUS_ROTATION,
            estimated_translation=NO_CONSENSUS_TRANSLATION,
        )
    consensus = _get_estimated_translation_rotation(
        [cells[i] for i in inlier_ids], reference_center
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
    criterion.

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
            current_ids, criterion_current = _refine(
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
    """
    Update the cell.is_congruent property.

    :param cells: list of cells.
    :param congruent_ids: list of cell ids that are congruent
    """

    for i, cell in enumerate(cells):
        cell.is_congruent = i in set(congruent_ids)


def _get_estimated_translation_rotation(
    cells: list[Cell], reference_center: tuple[float, float]
) -> CMCTranslationRotation:
    """
    Calculate shared rotation and transformation.

    :param cells: list of cells to fit.
    :param reference_center: reference center
    :returns: shared rotation and transformation, in CMCTranslationRotation
    """
    if len(cells) > 1:
        consensus_parameters = find_consensus_parameters(cells)
        # Negate to match the angle convention of Cell.angle_deg.
        consensus_rotation_deg = -float(np.degrees(consensus_parameters.rotation_rad))
        consensus_translation = get_translation_about(
            consensus_parameters, reference_center
        )
    else:
        # There is only one cell to fit
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
) -> tuple[list[int], float]:
    """
    Iteratively re-fit inlier set and criterion.

    :param current_ids: a list of inlier indices (used for least-squares Procrustes fit)
    :param criterion_current: the current value of the criterion
    :param cells: a list of cells
    :param max_distance: maximum distance threshold (meters)
    :param max_abs_angle_distance: maximum absolute angle threshold (degrees)
    :returns: tuple of (updated inlier indices, updated criterion)
    """
    best_ids = current_ids
    best_criterion = criterion_current

    while True:
        cell_distances, cell_angle_distances = _get_cell_angle_and_position_distances(
            best_ids, cells
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

        # Accept if strictly more inliers, or same count with lower criterion
        if len(candidate_ids) > len(best_ids) or (
            len(candidate_ids) == len(best_ids) and criterion_candidate < best_criterion
        ):
            best_ids = candidate_ids
            best_criterion = criterion_candidate
        else:
            # Local optimum reached
            return best_ids, best_criterion
