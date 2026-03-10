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
        Estimate a rigid body transformation (rotation + translation) from just those two cells via eval_cmc_consensus → point_register
        Find all other cells that fall within disMax and angleMax of that predicted location
        Attempt to iteratively refine by re-fitting using all successful cells (a least-squares improvement step)
        Keep the solution if it yields more CMC cells than the current best (or equal count with better quality)
     3. Get a boolean vector flagging which cells are CMC
     4. Return a ComparisonResult

     :param cells: Per-cell registration results to classify.
     :param params: Algorithm parameters (thresholds for score, angle, and position).
     :param reference_center: Global center [x, y] of the reference surface in meters,
         used as the fixed point for the median rotation.
     :returns: A :class:`ComparisonResult` containing the classified cells, median
         rotation in degrees, and median translation in meters.
     :raises ValueError: If ``cells`` is empty.
     """
    if not cells:
        raise ValueError("Cannot identify CMC from an empty list.")

    filtered_cells  = _filter_cells(cells, params.minimum_fill_fraction)
    n_filtered_cells = len(filtered_cells)


    max_distance = params.position_threshold # in meters
    max_abs_angle_distance = params.angle_deviation_threshold # in degrees

    # initialize solution
    n_congruent_cells = 1
    criterion = np.inf
    inliers_idx_previous = []
    criterion_previous = np.inf

    for idx_1, cell_1 in enumerate(filtered_cells[-1]):
        filtered_cells_gt_idx = filtered_cells[idx_1 + 1:]
        for idx_2, cell_2 in filtered_cells_gt_idx:
            cell_distances, cell_angle_distances = _get_cmc_consensus([idx_1, idx_1 + idx_2 + 1], filtered_cells)
            inliers_idx_new = np.where((cell_distances <= max_distance) & (cell_angle_distances <= max_abs_angle_distance))[0].tolist()
            criterion_previous = _calculate_criterion(inliers_idx_new, filtered_cells, max_distance, max_abs_angle_distance)
            if len(inliers_idx_new > 2) & len(inliers_idx_new < n_filtered_cells):
                # try to improve through least squares fit of all successful cells

                inliers_idx_previous = inliers_idx_new
                while len(inliers_idx_new) >= len(inliers_idx_previous):
                    cell_distances, cell_angle_distances = _get_cmc_consensus(inliers_idx_new,
                                                                              filtered_cells)
                    inliers_idx_new = \
                    np.where((cell_distances <= max_distance) & (cell_angle_distances <= max_abs_angle_distance))[
                        0].tolist()


                    criterion_new = _calculate_criterion(inliers_idx_new, filtered_cells, max_distance,
                                                              max_abs_angle_distance)
                    # check if new solution is better
                    if (len(inliers_idx_new) > len(inliers_idx_previous)) | ((len(inliers_idx_new) == len(inliers_idx_previous)) & (criterion_new < criterion_previous)):

                        # we have a better solution

                        criterion_previous = criterion_new
                        inliers_idx_previous = inliers_idx_new

                    else:

                        #stop trying to improve

                        inliers_idx_new = []

            # check if new solution is better

            if (len(inliers_idx_previous) > n_congruent_cells) | ((len(inliers_idx_previous) == n_congruent_cells) & (criterion_previous < criterion)):
                n_congruent_cells = len(inliers_idx_previous)
                criterion = criterion_previous


def _filter_cells(
    cells: list[Cell], minimum_fill_fraction: float
) -> list[Cell]:
    pass

def _calculate_criterion(cell_ids: list[int], all_cells: list[Cell], max_distance: float, max_abs_angle_distance: float) -> float:
    pass

def _get_cmc_consensus(included_idx: list[int], cells: list[Cell]) -> tuple[list[float], list[float]]:
    """

    """
    cells_for_fitting = [cells[idx] for idx in included_idx]
    consensus_translation, consensus_rotation_reference = _find_consensus_parameters(cells_for_fitting)
    cell_distances, cell_angle_distances = _get_distances(cells, consensus_translation, consensus_rotation_reference)

    return cell_distances, cell_angle_distances


def _find_consensus_parameters(cells: list[Cell]) -> tuple[list[float], list[float]]:
    # least squares fit to cells rotation and translation to find consensus parameters

    return consensus_translation, consensus_rotation_reference


def _get_distances(
    cells: list[Cell], consensus_translation: list[float], consensus_rotation_reference: list[float]
) -> tuple[list[float], list[float]]:

    return cell_distances, cell_angle_distances


