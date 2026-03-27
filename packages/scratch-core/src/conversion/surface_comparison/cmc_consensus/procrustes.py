from typing import Sequence

import numpy as np

from container_models.base import FloatArray2D, FloatArray1D
from conversion.surface_comparison.models import (
    Cell,
)

from conversion.surface_comparison.cmc_consensus.models import (
    ConsensusParameters,
)


def find_consensus_parameters(
    cells: Sequence[Cell],
) -> ConsensusParameters:
    """Least-squares 'Procrustes' rotation fit to find consensus rotation and translation parameters.

    Explanation of the method:
    Say we have two coordinate-pair lists [X] and [Y] where X_i is coupled with Y_i. And we want to find the rotation matrix R and translation of X to Y for which:
    ||(X - rotation_center_X) R - (Y - rotation_center_Y)||F_2 is minimal. i.e. the Frobenius norm (in this application the sum of squared distances between the linearly transformed set of points and the target set of points) is minimal.
    The rotation_centra yielding minimum Frobenius norm for the rotation operation are the coordinate means of X and Y. Note that this is a different coordinate system than the one used to find the rotation of individual cells during registration but this does not matter since optimal rotation angle is independent of coordinate system. We just want to find this rotation by minimizing the Frobenius norm and this centering minimizes the norm.
    It immediately follows that optimal translation in this coordinate system is rotation_center_Y - rotation_center_X.
    The optimal rotation can be found by completing the square in the Frobenius norm and observing that only the linear term
    -2trace(R^T X_centered^T Y_centered) depends on R. Therefore, this term should be minimal. Now, regard X_centered^T Y_centered = M with singular_value_decomposition(M) = U Sigma V^T, with U and V orthonormal basis and Sigma a diagonal eigenvalue matrix.
    For trace(R^T U Sigma V^T) to be maximal, since R, U and V are orthonormal matrices, you want: trace(R^T U Sigma V^T) = trace(Sigma). In order to achieve this (using the cyclic property of trace):
    trace(R^T U Sigma V^T) = trace(V^T R^T U Sigma), so R^T = V U^T, so R = U V^T.
    One last thing: since R is the collection of rotations and reflections, and physically we do not want reflections, we constrain the solution to reflections only by solving the above eigenvalue problem and, in case of reflection (determinant(R) = -1), reflecting the last axis of U (with the smallest eigenvalue, therefore yielding the minimal Frobenius norm given this contraint).

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
    cells: Sequence[Cell],
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
    cells: Sequence[Cell],
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
    data: np.ndarray, center: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate data around center, return only rotation component (no offset by center).

    :param data: data to be rotated, shape (n ,m), n cases with m features
    :param center: center of rotation, shape (1 ,m)
    :param rotation_matrix: rotation matrix, shape (m, m)

    :returns: rotated data minus center, shape (n ,m)
    """

    rotated = (data - center) @ rotation_matrix

    return rotated


def _get_rotation_component_using_angle_degree(
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

    return _get_rotation_component_using_rotation_matrix(
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
