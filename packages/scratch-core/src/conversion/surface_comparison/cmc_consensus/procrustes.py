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
    """
    Least-squares 'Procrustes' rotation fit to find consensus rotation and translation parameters.

    See README.md @ 'Explanation of Procrustes procedure' for details.

    :param cells: cells whose (center_reference, center_comparison) pairs are used for fitting
    :returns: consensus parameters: rotation centers (the centroids) and rotation angle in radians
    """
    reference_centers = np.array([cell.center_reference for cell in cells], dtype=float)
    comparison_centers = np.array(
        [cell.center_comparison for cell in cells], dtype=float
    )

    consensus_parameters = ConsensusParameters(
        rotation_center_reference=reference_centers.mean(axis=0),
        rotation_center_comparison=comparison_centers.mean(axis=0),
    )
    consensus_parameters.rotation_rad = float(
        get_rotation_angle_closed_form(
            centered_reference_centers=reference_centers
            - consensus_parameters.rotation_center_reference,
            centered_comparison_centers=comparison_centers
            - consensus_parameters.rotation_center_comparison,
        )
    )

    return consensus_parameters


def get_rotation_angle_closed_form(
    centered_reference_centers: np.ndarray, centered_comparison_centers: np.ndarray
) -> np.ndarray:
    """
    Closed-form 2-D Procrustes rotation angle (the best proper rotation, never a reflection).

    Works on a single point set of shape (k, 2) or on a batch of point sets of shape (..., k, 2).

    :param centered_reference_centers: reference centers with their centroid subtracted, shape (..., k, 2)
    :param centered_comparison_centers: comparison centers with their centroid subtracted, shape (..., k, 2)
    :returns: rotation angle(s) in radians, in [-pi, pi], shape (...)
    """
    cross_covariance = np.einsum(
        "...ki,...kj->...ij", centered_reference_centers, centered_comparison_centers
    )
    return np.arctan2(
        cross_covariance[..., 0, 1] - cross_covariance[..., 1, 0],
        cross_covariance[..., 0, 0] + cross_covariance[..., 1, 1],
    )


def get_translation_about(
    consensus_parameters: ConsensusParameters, center: tuple[float, float]
) -> tuple[float, float]:
    """
    Express the consensus translation around `center` instead of around the fit centroid.

    The Procrustes fit rotates around the cell centroids, so its translation only holds there.

    :param consensus_parameters: fitted rotation and rotation centers
    :param center: center to express the translation around, (x, y) meters
    :returns: translation from reference to comparison frame, (x, y) meters
    """
    rotated_center = get_rotation_component_using_angle_degree(
        xy_data=np.array(center),
        angle_deg=float(np.degrees(consensus_parameters.rotation_rad)),
        reference_center=consensus_parameters.rotation_center_reference,
    )[0]
    translation = (
        rotated_center + consensus_parameters.rotation_center_comparison - center
    )

    return float(translation[0]), float(translation[1])


def get_rotation_component_using_angle_degree(
    xy_data: FloatArray2D | FloatArray1D,
    angle_deg: float,
    reference_center: FloatArray1D,
) -> FloatArray2D:
    """
    Rotate data around a center, returning only the rotation component (the center is not added back).

    :param xy_data: data to be rotated, shape (n, 2) for n points, or (2,) for one point
    :param angle_deg: rotation angle in degrees
    :param reference_center: center of rotation, shape (2,)
    :returns: rotated data relative to the center, shape (n, 2)
    """
    return _get_rotation_component_using_rotation_matrix(
        data=xy_data,
        center=reference_center.reshape(1, -1),
        rotation_matrix=_build_2d_rotation_matrix(np.radians(angle_deg)),
    )


def _get_rotation_component_using_rotation_matrix(
    data: FloatArray2D | FloatArray1D,
    center: FloatArray2D,
    rotation_matrix: FloatArray2D,
) -> FloatArray2D:
    """
    Rotate data around a center, returning only the rotation component (the center is not added back).

    :param data: data to be rotated, shape (n, m) for n points with m coordinates, or (m,) for one point
    :param center: center of rotation, shape (1, m)
    :param rotation_matrix: rotation matrix for row vectors, shape (m, m)
    :returns: rotated data relative to the center, shape (n, m)
    """
    return (data - center) @ rotation_matrix


def _build_2d_rotation_matrix(angle_rad: float) -> FloatArray2D:
    """
    Build a 2-D rotation matrix for row vectors (`points @ matrix`).

    The matrix is [[cos, sin], [-sin, cos]], the transpose of the usual column-vector matrix, so that
    `[x, y] @ matrix` gives x' = x*cos - y*sin and y' = x*sin + y*cos: a counter-clockwise rotation by `angle_rad`.

    :param angle_rad: rotation angle in radians
    :returns: rotation matrix, shape (2, 2)
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([[cos_a, sin_a], [-sin_a, cos_a]])
