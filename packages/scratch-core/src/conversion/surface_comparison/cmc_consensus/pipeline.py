import numpy as np

from conversion.surface_comparison.cmc_consensus.criterion import calculate_criterion
from conversion.surface_comparison.cmc_consensus.models import (
    CMCTranslationRotation,
)
from conversion.surface_comparison.cmc_consensus.procrustes import (
    _get_rotation_component_using_angle_degree,
    find_consensus_parameters,
    rotation_angle_closed_form,
    get_translation_about,
)
from conversion.surface_comparison.models import (
    Cell,
    ComparisonResult,
)
from conversion.surface_comparison.utils import rotate_points, wrap_angles
from collections.abc import Callable

_PAIR_CHUNK = 1024
Fit = Callable[[list[int]], tuple[list[int], float]]


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
            cells, position_threshold, angle_deviation_threshold
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
        [cells[i] for i in inlier_ids], reference_center
    )
    # Apply the similarity threshold to the inliers
    cmc_ids = [i for i in inlier_ids if cells[i].best_score >= correlation_threshold]
    _update_congruent_cells(cells, cmc_ids)
    _update_cell_meta_data(cells, inlier_ids, consensus, reference_center)

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
    ref, comp, angle_rad = _cell_arrays(cells)
    fit = _make_fit(cells, max_distance, max_abs_angle_distance)

    best_ids: list[int] = []
    best_criterion = np.inf
    first, second = np.triu_indices(
        n_cells, k=1
    )  # same order as itertools.combinations

    for start in range(0, len(first), _PAIR_CHUNK):
        i, j = first[start : start + _PAIR_CHUNK], second[start : start + _PAIR_CHUNK]
        # Two-point fit: centroids are the midpoints
        center_ref, center_comp = (ref[i] + ref[j]) / 2, (comp[i] + comp[j]) / 2
        ref_pair = np.stack([ref[i] - center_ref, ref[j] - center_ref], axis=1)
        comp_pair = np.stack([comp[i] - center_comp, comp[j] - center_comp], axis=1)
        rotation = rotation_angle_closed_form(ref_pair, comp_pair)

        d, a = _residuals(ref, comp, angle_rad, rotation, center_ref, center_comp)
        inlier_mask = (d <= max_distance) & (a <= max_abs_angle_distance)
        counts = inlier_mask.sum(axis=1)

        for row in range(len(i)):
            count = int(counts[row])
            # A seed with <= 2 inliers is not refined, so it can only win while the best has <= 2 cells
            if count <= 2 and count < len(best_ids):
                continue
            current_ids = np.flatnonzero(inlier_mask[row]).tolist()
            current_criterion = calculate_criterion(
                d[row, current_ids],
                a[row, current_ids],
                max_distance,
                max_abs_angle_distance,
            )
            if 2 < count < n_cells:
                current_ids, current_criterion = _refine(
                    current_ids, current_criterion, fit
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

    :param cells: list of cells.
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
    Calculate shared rotation and transformation.

    :param cells: list of cells
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
        # There is only one cell
        inlier_cell = cells[0]
        predicted_coordinate = list(
            _get_rotation_component_using_angle_degree(
                np.array(inlier_cell.center_reference),
                -inlier_cell.angle_deg,
                np.array(reference_center),
            )[0]
            + np.array(reference_center)
        )
        consensus_translation = (
            float(inlier_cell.center_comparison[0] - predicted_coordinate[0]),
            float(inlier_cell.center_comparison[1] - predicted_coordinate[1]),
        )
        consensus_rotation_deg = inlier_cell.angle_deg

    shared_parameters = CMCTranslationRotation(
        translation=consensus_translation, rotation=consensus_rotation_deg
    )

    return shared_parameters


def _residuals(ref, comp, angle_rad, rotation, center_ref, center_comp):
    """Distances (m) and |angle| residuals (deg) of all cells for P fits -> two (P, n) arrays."""
    cos, sin = np.cos(rotation)[:, None], np.sin(rotation)[:, None]
    dx = ref[None, :, 0] - center_ref[:, 0:1]
    dy = ref[None, :, 1] - center_ref[:, 1:2]
    # Same convention as `(data - center) @ _build_2d_rotation_matrix(rotation)`
    pred_x = dx * cos - dy * sin + center_comp[:, 0:1]
    pred_y = dx * sin + dy * cos + center_comp[:, 1:2]
    distances = np.sqrt(
        (comp[None, :, 0] - pred_x) ** 2 + (comp[None, :, 1] - pred_y) ** 2
    )
    # Cell angles are in pixel coordinates and the fit in math coordinates, hence the +
    angle_distances = np.abs(
        np.degrees(wrap_angles(angle_rad[None, :] + rotation[:, None]))
    )
    return distances, angle_distances


def _cell_arrays(cells: list[Cell]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference centers (n, 2), comparison centers (n, 2) and cell angles in radians (n,)."""
    return (
        np.array([cell.center_reference for cell in cells], dtype=float),
        np.array([cell.center_comparison for cell in cells], dtype=float),
        np.radians([cell.angle_deg for cell in cells]),
    )


def _make_fit(
    cells: list[Cell], max_distance: float, max_abs_angle_distance: float
) -> Fit:
    """
    Build a memoized fit: Procrustes fit on the given ids, returning the inliers among all cells and their criterion.

    :param cells: list of cells.
    :param max_distance: maximum distance threshold (meters)
    :param max_abs_angle_distance: maximum absolute angle threshold (degrees)
    :returns: function mapping cell ids to (inlier ids, criterion)
    """
    ref, comp, angle_rad = _cell_arrays(cells)
    cache: dict[tuple[int, ...], tuple[list[int], float]] = {}

    def fit(ids: list[int]) -> tuple[list[int], float]:
        key = tuple(ids)
        if key not in cache:
            ids = list(key)
            center_ref, center_comp = ref[ids].mean(axis=0), comp[ids].mean(axis=0)
            rotation = rotation_angle_closed_form(
                ref[ids] - center_ref, comp[ids] - center_comp
            )
            d, a = _residuals(
                ref,
                comp,
                angle_rad,
                np.atleast_1d(rotation),
                center_ref[None],
                center_comp[None],
            )
            d, a = d[0], a[0]
            inliers = np.flatnonzero(
                (d <= max_distance) & (a <= max_abs_angle_distance)
            ).tolist()
            cache[key] = (
                inliers,
                calculate_criterion(
                    d[inliers], a[inliers], max_distance, max_abs_angle_distance
                ),
            )
        return cache[key]

    return fit


def _refine(
    current_ids: list[int], criterion_current: float, fit: Fit
) -> tuple[list[int], float]:
    """
    Iteratively re-fit inlier set and criterion until a local optimum is reached.

    :param current_ids: a list of inlier indices (used for least-squares Procrustes fit)
    :param criterion_current: the current value of the criterion
    :param fit: memoized fit from `_make_fit`
    :returns: tuple of (updated inlier indices, updated criterion)
    """
    best_ids, best_criterion = current_ids, criterion_current
    while True:
        candidate_ids, criterion_candidate = fit(best_ids)
        # Accept if strictly more inliers, or same count with lower criterion
        if len(candidate_ids) > len(best_ids) or (
            len(candidate_ids) == len(best_ids) and criterion_candidate < best_criterion
        ):
            best_ids, best_criterion = candidate_ids, criterion_candidate
        else:
            return best_ids, best_criterion
