import numpy as np

from container_models.base import FloatArray2D, FloatArray1D
from conversion.surface_comparison.models import (
    Cell,
)

from conversion.surface_comparison.cmc_consensus.models import (
    ConsensusParameters,
)


def find_consensus_parameters(
    cells: list[Cell],
) -> ConsensusParameters:
    """Least-squares 'Procrustes' rotation fit to find consensus rotation and translation parameters.

    See README.md @ 'Explanation of Procrustes procedure' for details.

    :param cells: cells whose (center_reference, center_comparison) pairs are used for fitting
    :returns: consensus_parameters
    """
    consensus_parameters = _get_translation(cells)

    centers_reference_centered, centers_comparison_centered = _center(
        cells,
        consensus_parameters.rotation_center_reference,
        consensus_parameters.rotation_center_comparison,
    )

    consensus_parameters.rotation_rad = _get_rotation_angle(
        centers_reference_centered, centers_comparison_centered
    )

    return consensus_parameters


def _get_translation(
    cells: list[Cell],
) -> ConsensusParameters:
    """
    :param cells: cells whose (center_reference, center_comparison) are used for fitting
    :returns: ConsensusParameters filled with translation part
    """
    centers_reference = np.array(
        [cell.center_reference for cell in cells], dtype=float
    )  # (N, 2)
    centers_comparison = np.array(
        [cell.center_comparison for cell in cells], dtype=float
    )  # (N, 2)

    reference_mean = centers_reference.mean(axis=0)  # mean of reference positions (2,)

    comparison_mean = centers_comparison.mean(
        axis=0
    )  # mean of comparison positions (2,)

    return ConsensusParameters(
        rotation_center_reference=reference_mean,
        rotation_center_comparison=comparison_mean,
    )


def _center(
    cells: list[Cell],
    rotation_center_reference: FloatArray1D,
    rotation_center_comparison: FloatArray1D,
) -> tuple[FloatArray2D, FloatArray2D]:
    """
    :param cells: cells whose (center_reference, center_comparison) are used for fitting
    :param rotation_center_reference: rotation_center_reference
    :param rotation_center_comparison: rotation_center_comparison

    :returns: centers with respective means substracted
    """
    centers_reference = np.array([cell.center_reference for cell in cells])
    centers_comparison = np.array([cell.center_comparison for cell in cells])

    centers_reference_centered = (
        centers_reference - rotation_center_reference
    )  # centred reference positions
    centers_comparison_centered = (
        centers_comparison - rotation_center_comparison
    )  # centred comparison positions

    return centers_reference_centered, centers_comparison_centered


def _get_rotation_angle(
    centers_reference_centered: FloatArray2D, centers_comparison_centered: FloatArray2D
) -> float:
    """
    :param centers_reference_centered: centers_reference with cells_mean subtracted
    :param centers_comparison_centered: centers_comparison with cells_mean subtracted
    :return: consensus_rotation_rad
    """
    # SVD for best-fit rotation (no reflection)

    cross_terms_matrix = (
        centers_reference_centered.T @ centers_comparison_centered
    )  # (2, 2)
    left_singular_vectors, _, right_singular_vectors_transposed = np.linalg.svd(
        cross_terms_matrix
    )
    rotation_matrix = (
        left_singular_vectors @ right_singular_vectors_transposed
    )  # (2, 2)

    # If det == -1 we have an unintended reflection; correct it by changing sign of last column of left_singular_vectors.
    if np.linalg.det(rotation_matrix) == -1:
        left_singular_vectors[:, -1] *= -1
        rotation_matrix = left_singular_vectors @ right_singular_vectors_transposed

    # Rotation angle: atan2(sin/cos)
    (cos, sin) = tuple(rotation_matrix[0])
    consensus_rotation_rad = float(np.arctan2(sin, cos))  # -pi <= angle <= pi

    return consensus_rotation_rad


def _get_rotation_component_using_rotation_matrix(
    data: FloatArray2D | FloatArray1D,
    center: FloatArray2D,
    rotation_matrix: FloatArray2D,
) -> FloatArray2D:
    """Rotate data around center, return only rotation component (no offset by center).

    :param data: data to be rotated, shape (n ,m), n cases with m features, or (m,) for one case
    :param center: center of rotation, shape (1 ,m)
    :param rotation_matrix: rotation matrix, shape (m, m)

    :returns: rotated data minus center, shape (n ,m)
    """

    rotated = (data - center) @ rotation_matrix

    return rotated


def _get_rotation_component_using_angle_degree(
    xy_data: FloatArray2D, angle_deg: float, reference_center: FloatArray2D
) -> FloatArray2D:
    """Rotate data around center.

    :param xy_data: data to be rotated, shape (n ,2), n cases with 2 features, or (2,) for 1 case
    :param angle_deg: angle in degrees
    :param reference_center: center of rotation, shape (2,)
    :returns rotated data, shape (n ,2)
    """
    reference_center = reference_center.reshape(1, -1)
    angle_rad = np.radians(angle_deg)
    rotation_matrix = _build_2d_rotation_matrix(angle_rad)

    return _get_rotation_component_using_rotation_matrix(
        data=xy_data, center=reference_center, rotation_matrix=rotation_matrix
    )


def _build_2d_rotation_matrix(angle_rad: float) -> FloatArray2D:
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
