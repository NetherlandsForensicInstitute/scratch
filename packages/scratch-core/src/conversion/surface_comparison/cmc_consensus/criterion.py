import numpy as np

from container_models.base import FloatArray2D, FloatArray1D
from conversion.surface_comparison.cmc_consensus.procrustes import (
    find_consensus_parameters,
    _build_2d_rotation_matrix,
    _get_rotation_component_using_rotation_matrix,
)
from conversion.surface_comparison.models import Cell


def calculate_criterion(
    cell_distances: FloatArray1D,
    cell_angle_distances: FloatArray1D,
    max_distance: float,
    max_abs_angle_distance: float,
) -> float:
    """Calculate criterion: mean_normalized_distance + mean_normalized_angle_distance.

    :param cell_distances: array of distances between cells and predicted positions (meters)
    :param cell_angle_distances: array of absolute angle distances (degrees)
    :param max_distance: maximum distance threshold (meters)
    :param max_abs_angle_distance: maximum absolute angle threshold (degrees)
    :returns: criterion (float)
    """
    if cell_distances.size == 0:
        return np.inf

    criterion = (
        float(np.mean(cell_distances)) / max_distance
        + float(np.mean(cell_angle_distances)) / max_abs_angle_distance
    )
    return criterion


def _get_cell_angle_and_position_distances(
    included_ids: list[int],
    cells: list[Cell],
) -> tuple[FloatArray1D, FloatArray1D]:
    """
    Calculate cell distances and cell angle distances after finding consensus parameters.

    :param included_ids: a list of included cell indices (used for least-squares fit)
    :param cells: a list of all filtered cells
    :param reference_rotation_center: global center [x, y] used as the fixed rotation point (meters)
    :returns: cell_distances (meters), cell_angle_distances in absolute degrees — both as FloatArray1D of length len(cells)
    """
    cells_for_least_squares = [cells[ids] for ids in included_ids]
    consensus_parameters = find_consensus_parameters(cells_for_least_squares)
    cell_distances, cell_angle_distances = _get_distances(
        cells,
        consensus_parameters.rotation_rad,
        rotation_center_reference=consensus_parameters.rotation_center_reference,
        rotation_center_comparison=consensus_parameters.rotation_center_comparison,
    )
    return cell_distances, cell_angle_distances


def _get_distances(
    cells: list[Cell],
    consensus_rotation_rad: float,
    rotation_center_reference: FloatArray1D,
    rotation_center_comparison: FloatArray1D,
) -> tuple[FloatArray1D, FloatArray1D]:
    """Get distances and abs_angle_distances of cell locations/rotations versus
    consensus_translation and consensus_rotation_rad.

    :param cells: a list of cells
    :param consensus_rotation_rad: rotation in radians (used to predict comparison positions
        AND to compute angle residuals after converting to degrees)
    :param rotation_center_reference: center of rotation in reference frame, shape (1 ,m)
    :param rotation_center_comparison: center of rotation in comparison frame, shape (1 ,m)
    :returns: distances (meters) as FloatArray1D, abs_angle_distances (unsigned degrees) as FloatArray1D
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


def _predict_positions(
    cells: list[Cell],
    consensus_rotation_rad: float,
    rotation_center_reference: FloatArray1D,
    rotation_center_comparison: FloatArray1D,
) -> FloatArray2D:
    """Predict positions of cells from reference_centers in comparison frame after rotation and translation by consensus values.

    :param cells: a list of cells
    :param consensus_rotation_rad: rotation angle in radians
    :param rotation_center_reference: (x, y) in meters
    :param rotation_center_comparison: (x, y) in meters
    :returns: predicted_positions list of (x, y) in meters
    """

    rotation_matrix = _build_2d_rotation_matrix(consensus_rotation_rad)
    rotation_center = rotation_center_reference.reshape(1, 2)
    cell_centers_reference = np.array(
        [cell.center_reference for cell in cells]
    )  # (n, 2)
    references_rotated = _get_rotation_component_using_rotation_matrix(
        data=cell_centers_reference,
        center=rotation_center,
        rotation_matrix=rotation_matrix,
    )

    # predicted position in comparison frame = rotation + center
    predicted_positions = references_rotated + rotation_center_comparison

    return predicted_positions


def _get_distances_meters(
    cells: list[Cell],
    predicted_positions: FloatArray2D,
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
