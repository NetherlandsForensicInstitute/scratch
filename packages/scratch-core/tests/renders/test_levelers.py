import numpy as np
import pytest

from container_models.base import Point
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms
from renders.levelers import level_map

# WARNING: AI generated
# This was promted with:
# generate renders.levelers.level_map tests by referncing @tests/conversion/leveling/solvers/test_core.py
# Here's the mapping of tests in test_levelers.py that reference concepts from test_core.py:
# Direct References/Adaptations:
#
# 1. Variance Reduction Pattern
#
# - test_core.py:11 - test_fit_surface_reduces_variance
#   - Checks: var(fitted_surface) < var(zs) and var(fitted_surface) < var(leveled_map)
# - test_levelers.py:88 - test_reduces_variance_for_tilted
#   - Adapted to check: leveled_var < original_var (same concept, different API)
#
# 2. SPHERE vs Other Terms Comparison
#
# - test_core.py:42 - test_fit_surface_sphere_reduces_variance
#   - Checks: var(leveled with SPHERE) < var(leveled with other terms)
# - test_levelers.py:138 - test_sphere_better_than_plane_for_curved
#   - Same concept: compares SPHERE vs PLANE on curved surface
#
# 3. OFFSET Equals Mean
#
# - test_core.py:83 - test_fit_surface_offset_equals_mean
#   - Checks: OFFSET parameter equals mean, fitted surface is constant
# - test_levelers.py:121 - test_offset_centers_data_at_zero
#   - Adapted: After leveling with OFFSET, mean should be ~0
#
# 4. NONE Has No Effect
#
# - test_core.py:103 - test_fit_surface_none_has_no_effect
#   - Checks: fitted_surface == 0 and all params are 0
# - test_levelers.py:111 - test_none_has_no_effect
#   - Checks: leveled data equals original data (equivalent concept)
#
# 5. PLANE Comparison
#
# - test_core.py:23 - test_fit_surface_plane_reduces_variance
#   - Compares single terms vs PLANE
# - test_levelers.py:129 - test_plane_levels_tilted_surface
#   - Tests PLANE effectiveness on tilted surface (variance near 0)
#
# Additional Tests Not in test_core.py:
#
# The following tests in test_levelers.py are unique and don't have direct references in test_core.py:
# - Lines 37-75: TestLevelMapBasic (shape, NaN handling, dtype checks)
# - Lines 156-186: TestLevelMapReferencePoint (reference point invariance)
# - Lines 189-219: TestLevelMapWithNaNs (NaN-specific edge cases)
# - Lines 222-279: TestLevelMapIntegration (high-pass filter behavior, workflow tests)


@pytest.fixture
def tilted_scan_image_with_nans() -> ScanImage:
    """Create a tilted scan image with some NaN values."""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)
    data = 2.0 * X + 3.0 * Y + 1.0
    # Add some NaN values
    data[5, 5] = np.nan
    data[10, 10] = np.nan
    data[15, 15] = np.nan
    return ScanImage(data=data, scale_x=0.1, scale_y=0.1)


@pytest.fixture
def curved_scan_image() -> ScanImage:
    """Create a scan image with a curved (quadratic) surface."""
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    data = X**2 + Y**2 + 2.0 * X + 3.0 * Y + 1.0
    return ScanImage(data=data, scale_x=0.1, scale_y=0.1)


class TestLevelMapBasic:
    def test_output_shape_matches_input(self, simple_scan_image: ScanImage):
        """Test that leveled map has the same shape as input scan image."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(simple_scan_image, SurfaceTerms.PLANE, reference_point)

        assert leveled.shape == simple_scan_image.data.shape

    def test_preserves_nan_locations(self, simple_scan_image: ScanImage):
        """Test that NaN values remain in the same positions after leveling."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(simple_scan_image, SurfaceTerms.PLANE, reference_point)

        # NaN should be in the same locations
        np.testing.assert_array_equal(
            np.isnan(leveled), np.isnan(simple_scan_image.data)
        )

    def test_valid_data_is_not_nan(self, simple_scan_image: ScanImage):
        """Test that valid data points are not NaN after leveling."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(simple_scan_image, SurfaceTerms.PLANE, reference_point)

        # Where original data was valid, leveled should also be valid
        assert not np.any(np.isnan(leveled[simple_scan_image.valid_mask]))

    def test_returns_2d_array(self, simple_scan_image: ScanImage):
        """Test that the function returns a 2D numpy array."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(simple_scan_image, SurfaceTerms.PLANE, reference_point)

        assert isinstance(leveled, np.ndarray)
        assert leveled.ndim == 2

    def test_dtype_is_float(self, simple_scan_image: ScanImage):
        """Test that output array has float dtype."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(simple_scan_image, SurfaceTerms.PLANE, reference_point)

        assert np.issubdtype(leveled.dtype, np.floating)


class TestLevelMapTerms:
    @pytest.mark.parametrize(
        "terms",
        [
            SurfaceTerms.TILT_X,
            SurfaceTerms.TILT_Y,
            SurfaceTerms.PLANE,
            SurfaceTerms.SPHERE,
        ],
    )
    def test_reduces_variance_for_tilted(
        self, tilted_scan_image: ScanImage, terms: SurfaceTerms
    ):
        """Test that appropriate terms reduce variance for a tilted surface."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(tilted_scan_image, terms, reference_point)

        original_var = np.var(tilted_scan_image.valid_data)
        leveled_var = np.var(leveled[tilted_scan_image.valid_mask])

        assert leveled_var < original_var

    def test_offset_reduces_variance_for_uniform(self, uniform_scan_image: ScanImage):
        """Test that OFFSET term works on uniform data."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(uniform_scan_image, SurfaceTerms.OFFSET, reference_point)

        original_var = np.var(uniform_scan_image.valid_data)
        leveled_var = np.var(leveled[uniform_scan_image.valid_mask])

        # For uniform data with constant offset, leveling should reduce to near zero
        assert leveled_var < original_var or np.isclose(leveled_var, 0.0, atol=1e-20)

    def test_none_has_no_effect(self, simple_scan_image: ScanImage):
        """Test that NONE term produces output equal to input."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(simple_scan_image, SurfaceTerms.NONE, reference_point)

        # Valid data should be unchanged
        np.testing.assert_allclose(
            leveled[simple_scan_image.valid_mask], simple_scan_image.valid_data
        )

    def test_offset_centers_data_at_zero(self, uniform_scan_image: ScanImage):
        """Test that OFFSET term centers the data around zero mean."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(uniform_scan_image, SurfaceTerms.OFFSET, reference_point)

        leveled_mean = np.mean(leveled[uniform_scan_image.valid_mask])
        assert np.isclose(leveled_mean, 0.0, atol=1e-10)

    def test_plane_levels_tilted_surface(self, tilted_scan_image: ScanImage):
        """Test that PLANE term effectively removes tilt from a tilted surface."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(tilted_scan_image, SurfaceTerms.PLANE, reference_point)

        # After leveling a perfect plane, the variance should be near zero
        leveled_var = np.var(leveled[tilted_scan_image.valid_mask])
        assert leveled_var < 1e-20  # Very small for a perfect fit

    def test_sphere_better_than_plane_for_curved(self, curved_scan_image: ScanImage):
        """Test that SPHERE terms fit curved surfaces better than PLANE."""
        reference_point = Point(0.0, 0.0)

        leveled_plane = level_map(
            curved_scan_image, SurfaceTerms.PLANE, reference_point
        )
        leveled_sphere = level_map(
            curved_scan_image, SurfaceTerms.SPHERE, reference_point
        )

        plane_var = np.var(leveled_plane[curved_scan_image.valid_mask])
        sphere_var = np.var(leveled_sphere[curved_scan_image.valid_mask])

        assert sphere_var < plane_var


class TestLevelMapReferencePoint:
    def test_different_reference_points_same_variance(
        self, tilted_scan_image: ScanImage
    ):
        """Test that different reference points produce similar variance reduction."""
        ref_points = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)]
        variances = []

        for ref_point in ref_points:
            leveled = level_map(tilted_scan_image, SurfaceTerms.PLANE, ref_point)
            variance = np.var(leveled[tilted_scan_image.valid_mask])
            variances.append(variance)

        # All reference points should produce similar variance
        assert np.std(variances) < 1e-20

    def test_reference_point_affects_coordinate_system_only(
        self, uniform_scan_image: ScanImage
    ):
        """Test that reference point doesn't affect the leveling result for uniform data."""
        ref_point_1 = Point(0.0, 0.0)
        ref_point_2 = Point(5.0, 5.0)

        leveled_1 = level_map(uniform_scan_image, SurfaceTerms.OFFSET, ref_point_1)
        leveled_2 = level_map(uniform_scan_image, SurfaceTerms.OFFSET, ref_point_2)

        # Results should be identical for uniform data
        np.testing.assert_allclose(
            leveled_1[uniform_scan_image.valid_mask],
            leveled_2[uniform_scan_image.valid_mask],
            rtol=1e-10,
        )


class TestLevelMapWithNaNs:
    def test_handles_scan_with_nans(self, tilted_scan_image_with_nans: ScanImage):
        """Test that leveling works correctly with NaN values in the scan."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(
            tilted_scan_image_with_nans, SurfaceTerms.PLANE, reference_point
        )

        # Output should have same shape
        assert leveled.shape == tilted_scan_image_with_nans.data.shape

        # NaN positions should be preserved
        np.testing.assert_array_equal(
            np.isnan(leveled), np.isnan(tilted_scan_image_with_nans.data)
        )

        # Valid data should be leveled (low variance)
        leveled_var = np.var(leveled[tilted_scan_image_with_nans.valid_mask])
        assert leveled_var < 1e-20

    def test_nan_values_do_not_affect_fit(self, tilted_scan_image_with_nans: ScanImage):
        """Test that NaN values don't influence the surface fit."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(
            tilted_scan_image_with_nans, SurfaceTerms.PLANE, reference_point
        )

        # The fit should still work well on valid data
        valid_leveled = leveled[tilted_scan_image_with_nans.valid_mask]
        assert not np.any(np.isnan(valid_leveled))
        assert np.var(valid_leveled) < 1e-20  # Good fit


class TestLevelMapIntegration:
    def test_complete_workflow_simple(self, simple_scan_image: ScanImage):
        """Test complete leveling workflow with simple data."""
        reference_point = Point(0.5, 0.5)
        leveled = level_map(simple_scan_image, SurfaceTerms.PLANE, reference_point)

        # Basic sanity checks
        assert leveled.shape == simple_scan_image.data.shape
        assert np.sum(np.isnan(leveled)) == np.sum(np.isnan(simple_scan_image.data))

        # Leveling should reduce variance
        original_var = np.var(simple_scan_image.valid_data)
        leveled_var = np.var(leveled[simple_scan_image.valid_mask])
        assert leveled_var < original_var

    def test_high_pass_filter_behavior(self, tilted_scan_image: ScanImage):
        """Test that level_map acts as a high-pass filter (removes low-frequency trends)."""
        reference_point = Point(0.0, 0.0)

        # Add high-frequency noise to the tilted surface
        rng = np.random.default_rng(42)
        noisy_data = tilted_scan_image.data + rng.normal(
            0, 0.1, tilted_scan_image.data.shape
        )
        noisy_scan = ScanImage(data=noisy_data, scale_x=0.1, scale_y=0.1)

        leveled = level_map(noisy_scan, SurfaceTerms.PLANE, reference_point)

        # The leveled map should remove the low-frequency tilt
        # but preserve high-frequency noise
        # Mean should be near zero (tilt removed)
        assert np.abs(np.mean(leveled[noisy_scan.valid_mask])) < 0.1

        # Variance should be similar to the noise variance
        leveled_std = np.std(leveled[noisy_scan.valid_mask])
        assert 0.05 < leveled_std < 0.15  # Noise std was 0.1

    @pytest.mark.parametrize(
        "terms",
        [
            SurfaceTerms.PLANE,
            SurfaceTerms.SPHERE,
            SurfaceTerms.OFFSET | SurfaceTerms.TILT_X,
        ],
    )
    def test_combined_terms(self, curved_scan_image: ScanImage, terms: SurfaceTerms):
        """Test that combined terms work correctly."""
        reference_point = Point(0.0, 0.0)
        leveled = level_map(curved_scan_image, terms, reference_point)

        # Should successfully level the data
        assert leveled.shape == curved_scan_image.data.shape
        assert not np.any(np.isnan(leveled[curved_scan_image.valid_mask]))

        # Should reduce variance
        original_var = np.var(curved_scan_image.valid_data)
        leveled_var = np.var(leveled[curved_scan_image.valid_mask])
        assert leveled_var < original_var
