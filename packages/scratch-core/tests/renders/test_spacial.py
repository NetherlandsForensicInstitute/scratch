import numpy as np
import pytest
from numpy.typing import NDArray

from container_models.base import Point, PointCloud
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms
from renders.spacial import fit_surface, generate_point_cloud

# WARNING: AI generated


@pytest.fixture
def simple_scan_image() -> ScanImage:
    """Create a simple 5x5 scan image with some NaN values for testing."""
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, np.nan, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ]
    )
    return ScanImage(data=data, scale_x=0.1, scale_y=0.1)


@pytest.fixture
def uniform_scan_image() -> ScanImage:
    """Create a uniform scan image without NaN values."""
    data = np.ones((10, 10)) * 5.0
    return ScanImage(data=data, scale_x=0.1, scale_y=0.1)


@pytest.fixture
def tilted_scan_image() -> ScanImage:
    """Create a scan image with a tilted surface."""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)
    data = 2.0 * X + 3.0 * Y + 1.0
    return ScanImage(data=data, scale_x=0.1, scale_y=0.1)


@pytest.fixture
def simple_point_cloud() -> PointCloud:
    """Create a simple point cloud for testing."""
    vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return PointCloud(xs=vector, ys=vector, zs=vector)


@pytest.fixture
def lstsq_solver():
    """Least squares solver implementation for testing."""

    def solver(
        design_matrix: NDArray[np.float64], zs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        coefficients, *_ = np.linalg.lstsq(design_matrix, zs, rcond=None)
        return coefficients

    return solver


class TestGeneratePointCloud:
    def test_generates_correct_shape(self, simple_scan_image: ScanImage):
        """Test that point cloud has the correct number of valid points."""
        reference_point = Point(0.0, 0.0)
        point_cloud = generate_point_cloud(simple_scan_image, reference_point)

        n_valid = np.sum(simple_scan_image.valid_mask)
        assert len(point_cloud.xs) == n_valid
        assert len(point_cloud.ys) == n_valid
        assert len(point_cloud.zs) == n_valid

    def test_point_cloud_contains_valid_data_only(self, simple_scan_image: ScanImage):
        """Test that point cloud only contains valid (non-NaN) data."""
        reference_point = Point(0.0, 0.0)
        point_cloud = generate_point_cloud(simple_scan_image, reference_point)

        assert not np.any(np.isnan(point_cloud.xs))
        assert not np.any(np.isnan(point_cloud.ys))
        assert not np.any(np.isnan(point_cloud.zs))

    def test_z_values_match_valid_data(self, simple_scan_image: ScanImage):
        """Test that z values in point cloud match the scan image's valid data."""
        reference_point = Point(0.0, 0.0)
        point_cloud = generate_point_cloud(simple_scan_image, reference_point)

        expected_zs = simple_scan_image.valid_data
        np.testing.assert_array_equal(point_cloud.zs, expected_zs)

    def test_reference_point_zero_starts_at_origin(self, uniform_scan_image: ScanImage):
        """Test that reference point (0, 0) places the first pixel at the origin."""
        reference_point = Point(0.0, 0.0)
        point_cloud = generate_point_cloud(uniform_scan_image, reference_point)

        # With reference point at origin, coordinates should start at 0
        assert np.min(point_cloud.xs) >= -0.01
        assert np.min(point_cloud.ys) >= -0.01

    def test_reference_point_offset_shifts_coordinates(
        self, uniform_scan_image: ScanImage
    ):
        """Test that non-zero reference point shifts the coordinate system."""
        reference_point_zero = Point(0.0, 0.0)
        reference_point_offset = Point(1.0, 1.0)

        pc_zero = generate_point_cloud(uniform_scan_image, reference_point_zero)
        pc_offset = generate_point_cloud(uniform_scan_image, reference_point_offset)

        # The offset point cloud should be shifted by the negative of the reference point
        expected_x_shift = -1.0
        expected_y_shift = -1.0

        assert np.isclose(
            np.mean(pc_offset.xs) - np.mean(pc_zero.xs), expected_x_shift, atol=0.01
        )
        assert np.isclose(
            np.mean(pc_offset.ys) - np.mean(pc_zero.ys), expected_y_shift, atol=0.01
        )

    def test_returns_point_cloud_type(self, simple_scan_image: ScanImage):
        """Test that the function returns a PointCloud instance."""
        reference_point = Point(0.0, 0.0)
        point_cloud = generate_point_cloud(simple_scan_image, reference_point)

        assert isinstance(point_cloud, PointCloud)

    def test_preserves_scan_scale(self, simple_scan_image: ScanImage):
        """Test that the generated point cloud respects the scan image scale."""
        reference_point = Point(0.0, 0.0)
        point_cloud = generate_point_cloud(simple_scan_image, reference_point)

        # The spacing between points should reflect the scale
        x_spacing = simple_scan_image.scale_x
        unique_xs = np.unique(point_cloud.xs)

        if len(unique_xs) > 1:
            actual_spacing = np.min(np.diff(unique_xs))
            assert np.isclose(actual_spacing, x_spacing, rtol=0.01)


class TestFitSurface:
    def test_returns_correct_shape(self, simple_point_cloud: PointCloud, lstsq_solver):
        """Test that fitted surface has the same shape as input z values."""
        terms = SurfaceTerms.PLANE
        fitted_surface = fit_surface(simple_point_cloud, terms, lstsq_solver)

        assert fitted_surface.shape == simple_point_cloud.zs.shape

    def test_offset_only_returns_mean(
        self, simple_point_cloud: PointCloud, lstsq_solver
    ):
        """Test that fitting only offset term returns the mean of z values."""
        terms = SurfaceTerms.OFFSET
        fitted_surface = fit_surface(simple_point_cloud, terms, lstsq_solver)

        expected_mean = np.mean(simple_point_cloud.zs)
        assert np.allclose(fitted_surface, expected_mean)

    def test_none_returns_zeros(self, simple_point_cloud: PointCloud, lstsq_solver):
        """Test that fitting no terms returns zeros."""
        terms = SurfaceTerms.NONE
        fitted_surface = fit_surface(simple_point_cloud, terms, lstsq_solver)

        assert np.allclose(fitted_surface, 0.0)

    def test_perfect_plane_fit(self, lstsq_solver):
        """Test that a perfect planar surface is fitted exactly."""
        xs = np.linspace(-5, 5, 50)
        ys = np.linspace(-5, 5, 50)
        zs = 2.0 * xs + 3.0 * ys + 1.0  # Perfect plane: z = 2x + 3y + 1

        point_cloud = PointCloud(xs=xs, ys=ys, zs=zs)
        fitted_surface = fit_surface(point_cloud, SurfaceTerms.PLANE, lstsq_solver)

        # For a perfect plane, fitted surface should match the original z values
        np.testing.assert_allclose(fitted_surface, zs, rtol=1e-10)

    def test_sphere_terms_fit_curved_surface(self, lstsq_solver):
        """Test that sphere terms can fit a curved surface better than plane."""
        xs = np.linspace(-5, 5, 50)
        ys = np.linspace(-5, 5, 50)
        zs = xs**2 + ys**2 + 2.0 * xs + 3.0 * ys + 1.0  # Curved surface

        point_cloud = PointCloud(xs=xs, ys=ys, zs=zs)

        fitted_plane = fit_surface(point_cloud, SurfaceTerms.PLANE, lstsq_solver)
        fitted_sphere = fit_surface(point_cloud, SurfaceTerms.SPHERE, lstsq_solver)

        # Sphere fit should have lower residuals than plane fit
        residual_plane = np.var(zs - fitted_plane)
        residual_sphere = np.var(zs - fitted_sphere)

        assert residual_sphere < residual_plane

    def test_fit_reduces_variance(self, lstsq_solver):
        """Test that fitted surface has lower variance than leveled data for typical data."""
        rng = np.random.default_rng(42)
        xs = rng.uniform(-10, 10, 100)
        ys = rng.uniform(-10, 10, 100)
        zs = 2.0 * xs + 3.0 * ys + 1.0 + rng.normal(0, 0.1, 100)

        point_cloud = PointCloud(xs=xs, ys=ys, zs=zs)
        fitted_surface = fit_surface(point_cloud, SurfaceTerms.PLANE, lstsq_solver)

        leveled = zs - fitted_surface

        assert np.var(fitted_surface) > np.var(leveled)

    def test_output_is_float64(self, simple_point_cloud: PointCloud, lstsq_solver):
        """Test that the output is always float64."""
        terms = SurfaceTerms.PLANE
        fitted_surface = fit_surface(simple_point_cloud, terms, lstsq_solver)

        assert fitted_surface.dtype == np.float64

    def test_handles_different_term_combinations(
        self, simple_point_cloud: PointCloud, lstsq_solver
    ):
        """Test that different term combinations work correctly."""
        term_combinations = [
            SurfaceTerms.OFFSET,
            SurfaceTerms.TILT_X,
            SurfaceTerms.TILT_Y,
            SurfaceTerms.PLANE,
            SurfaceTerms.DEFOCUS,
            SurfaceTerms.SPHERE,
            SurfaceTerms.OFFSET | SurfaceTerms.TILT_X,
            SurfaceTerms.TILT_X | SurfaceTerms.TILT_Y,
        ]

        for terms in term_combinations:
            fitted_surface = fit_surface(simple_point_cloud, terms, lstsq_solver)
            assert fitted_surface.shape == simple_point_cloud.zs.shape
            assert not np.any(np.isnan(fitted_surface))


class TestIntegration:
    def test_generate_and_fit_workflow(
        self, tilted_scan_image: ScanImage, lstsq_solver
    ):
        """Test the complete workflow of generating point cloud and fitting surface."""
        reference_point = Point(0.5, 0.5)

        # Generate point cloud
        point_cloud = generate_point_cloud(tilted_scan_image, reference_point)

        # Fit surface
        fitted_surface = fit_surface(point_cloud, SurfaceTerms.PLANE, lstsq_solver)

        # Verify results
        assert len(fitted_surface) == len(point_cloud.zs)
        assert not np.any(np.isnan(fitted_surface))

        # The fitted surface should approximate the original tilted surface well
        residuals = point_cloud.zs - fitted_surface
        assert np.std(residuals) < 0.1  # Low residuals for a good fit

    def test_multiple_reference_points_consistency(
        self, tilted_scan_image: ScanImage, lstsq_solver
    ):
        """Test that different reference points produce consistent fits (relative to their coordinate systems)."""
        ref_points = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)]
        fitted_variances = []

        for ref_point in ref_points:
            pc = generate_point_cloud(tilted_scan_image, ref_point)
            fitted = fit_surface(pc, SurfaceTerms.PLANE, lstsq_solver)
            residuals = pc.zs - fitted
            fitted_variances.append(np.var(residuals))

        # All reference points should produce similar fit quality (similar residual variance)
        assert np.std(fitted_variances) < 0.01
