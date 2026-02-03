from typing import NamedTuple, Protocol
from container_models.base import FloatArray1D
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms
from conversion.leveling.solver.design import build_design_matrix
from conversion.leveling.solver.grid import get_2d_grid
from conversion.leveling.solver.transforms import normalize_coordinates
from mutations.base import ImageMutation
import numpy as np
from numpy.typing import NDArray


class PointCloud[T: FloatArray1D](NamedTuple):
    xs: T
    ys: T
    zs: T


class DesignMatrixSolver(Protocol):
    """Protocol for solver functions that compute polynomial coefficients from a design matrix and Z values."""

    def __call__(self, design_matrix: NDArray, zs: NDArray) -> NDArray: ...


class LevelMap(ImageMutation):
    def __init__(
        self, x_reference_point: float, y_reference_point: float, terms: SurfaceTerms
    ) -> None:
        self.x_reference_point = x_reference_point
        self.y_reference_point = y_reference_point
        self.terms = terms

    def solve_least_squares(
        self, design_matrix: NDArray[np.float64], zs: NDArray
    ) -> NDArray[np.floating]:
        """
        Solve the least squares problem to find polynomial coefficients.
        :param design_matrix: The design matrix constructed from polynomial terms.
        :param zs: The Z-values (height data) to fit.
        :returns: Array of polynomial coefficients.
        """
        (
            coefficients,
            *_,
        ) = np.linalg.lstsq(design_matrix, zs, rcond=None)
        return coefficients

    def _fit_surface(
        self, point_cloud: PointCloud, terms: SurfaceTerms, solver: DesignMatrixSolver
    ) -> NDArray[np.floating]:
        """
        Core solver: fits a surface to the point cloud.
        :param point_cloud: PointCloud containing the X, Y, and Z coordinates.
        :param terms: The surface terms to use in the polynomial fitting.
        :param solver: Solver function that computes coefficients from design matrix and Z values.
        :returns: 1D array containing the fitted surface values (zs).
        """
        normalized = normalize_coordinates(point_cloud.xs, point_cloud.ys)
        design_matrix = build_design_matrix(normalized.xs, normalized.ys, terms)
        coefficients = solver(design_matrix=design_matrix, zs=point_cloud.zs)
        return design_matrix @ coefficients

    def _generate_point_cloud(
        self, scan_image: ScanImage, x_reference_point: float, y_reference_point: float
    ) -> PointCloud:
        """
        Generate a 3D point cloud from a scan image with coordinates centered at a reference point.
        :param scan_image: The scan image containing the height data and mask.
        :param reference_point: Tuple (x, y) in physical coordinates to use as the origin.
        :returns: PointCloud containing the valid X, Y, and Z coordinates.
        """
        x_grid, y_grid = get_2d_grid(
            scan_image, offset=(-x_reference_point, -y_reference_point)
        )
        xs, ys, zs = (
            x_grid[scan_image.valid_mask],
            y_grid[scan_image.valid_mask],
            scan_image.valid_data,
        )
        return PointCloud(xs=xs, ys=ys, zs=zs)

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Compute the leveled map by fitting polynomial terms and subtracting them from the image data.
        This computation effectively acts as a high-pass filter on the image data.
        :param scan_image: The scan image containing the image data to level.
        :param terms: The surface terms to use in the polynomial fitting. Can be combined using bit-operators.
        :param reference_point: Tuple (X, Y) in physical coordinate space defining the origin for coordinate translation.
            The coordinates will be translated such that (X, Y) lies at the origin.
        :returns: 2D array containing the leveled scan data (original data minus fitted surface).
        """
        point_cloud = self._generate_point_cloud(
            scan_image=scan_image,
            y_reference_point=self.y_reference_point,
            x_reference_point=self.x_reference_point,
        )
        fitted_surface = self._fit_surface(
            point_cloud=point_cloud,
            terms=self.terms,
            solver=self.solve_least_squares,  # TODO: check what i need to do with this. it was logical with functional programming, in OOP is this not logical.
        )
        leveled_map_2d = np.full_like(scan_image.data, np.nan)
        leveled_map_2d[scan_image.valid_mask] = point_cloud.zs - fitted_surface

        scan_image.data = leveled_map_2d
        return scan_image
