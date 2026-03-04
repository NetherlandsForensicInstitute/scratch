"""
Unit tests for the private helper functions in cmc_classification.

Each test class covers one function. Tests are structured with
Arrange / Act / Assert comments and kept to 2–3 cases per function.
"""

import numpy as np


from conversion.surface_comparison.cmc_classification import (
    _circular_median,
    _get_esd_criterion,
    _get_threshold_criterion,
    _outliers_gesd,
    _rosner_critical_value,
    _rotate_points,
    _wrap_angles,
)


# ---------------------------------------------------------------------------
# _wrap_angles
# ---------------------------------------------------------------------------


class TestWrapAngles:
    """Tests for _wrap_angles: normalises any angle array to [-pi, pi]."""

    def test_angles_within_range_are_unchanged(self) -> None:
        """Angles already in [-pi, pi] must pass through unmodified."""
        # Arrange
        angles = np.array([0.0, np.pi / 2, -np.pi / 3])

        # Act
        result = _wrap_angles(angles)

        # Assert
        np.testing.assert_allclose(result, angles)

    def test_angles_outside_range_are_wrapped(self) -> None:
        """Angles beyond ±pi must be folded back into [-pi, pi]."""
        # Arrange
        angles = np.array([3 * np.pi, 3.5 * np.pi, 2 * np.pi])
        expected = np.array([-np.pi, -np.pi / 2, 0.0])

        # Act
        result = _wrap_angles(angles)

        # Assert
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# _circular_median
# ---------------------------------------------------------------------------


class TestCircularMedian:
    """Tests for _circular_median: computes the circular median of radian angles."""

    def test_tight_cluster_returns_cluster_centre(self) -> None:
        """A tight cluster of similar angles should return a value near their mean."""
        # Arrange
        angles = np.radians(np.array([10.0, 11.0, 10.5, 9.8, 10.2]))

        # Act
        result = _circular_median(angles)

        # Assert — result must be within 1° of the cluster centre
        assert abs(np.degrees(result) - 10.3) < 1.0

    def test_single_angle_returns_itself(self) -> None:
        """The circular median of a single value must equal that value."""
        # Arrange
        angle = np.array([np.pi / 4])

        # Act
        result = _circular_median(angle)

        # Assert
        np.testing.assert_allclose(result, np.pi / 4, atol=1e-12)

    def test_symmetric_around_zero_returns_near_zero(self) -> None:
        """Angles symmetric around zero should yield a median near zero."""
        # Arrange
        angles = np.radians(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))

        # Act
        result = _circular_median(angles)

        # Assert
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _rotate_points
# ---------------------------------------------------------------------------


class TestRotatePoints:
    """Tests for _rotate_points: rotates (N, 2) point arrays around a centre."""

    def test_zero_rotation_returns_original_points(self) -> None:
        """Rotating by zero radians must leave every point unchanged."""
        # Arrange
        points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        center = np.array([0.0, 0.0])

        # Act
        result = _rotate_points(points, angle=0.0, center=center)

        # Assert
        np.testing.assert_allclose(result, points, atol=1e-12)

    def test_quarter_turn_around_origin(self) -> None:
        """A 90° CCW rotation of [1, 0] around the origin should give [0, 1]."""
        # Arrange
        points = np.array([[1.0, 0.0]])
        center = np.array([0.0, 0.0])

        # Act
        result = _rotate_points(points, angle=np.pi / 2, center=center)

        # Assert
        np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-12)

    def test_rotation_around_non_origin_center(self) -> None:
        """Rotating a point 180° around itself must return the same point."""
        # Arrange
        points = np.array([[3.0, 4.0]])
        center = np.array([3.0, 4.0])

        # Act
        result = _rotate_points(points, angle=np.pi, center=center)

        # Assert
        np.testing.assert_allclose(result, points, atol=1e-12)


# ---------------------------------------------------------------------------
# _rosner_critical_value
# ---------------------------------------------------------------------------


class TestRosnerCriticalValue:
    """Tests for _rosner_critical_value: Rosner (1983) GESD critical value formula."""

    def test_fewer_than_two_remaining_returns_inf(self) -> None:
        """With fewer than 2 remaining observations the critical value must be inf."""
        # Arrange / Act / Assert
        assert _rosner_critical_value(n_remaining=1, alpha=0.05) == np.inf

    def test_typical_value_is_finite_and_positive(self) -> None:
        """For a typical sample size the critical value should be finite and positive."""
        # Arrange
        n_remaining = 10

        # Act
        result = _rosner_critical_value(n_remaining=n_remaining, alpha=0.05)

        # Assert
        assert np.isfinite(result) and result > 0.0

    def test_smaller_alpha_gives_larger_critical_value(self) -> None:
        """A stricter significance level (smaller alpha) must raise the critical value."""
        # Arrange
        n_remaining = 20

        # Act
        critical_value_lenient = _rosner_critical_value(n_remaining, alpha=0.10)
        critical_value_strict = _rosner_critical_value(n_remaining, alpha=0.01)

        # Assert
        assert critical_value_strict > critical_value_lenient


# ---------------------------------------------------------------------------
# _outliers_gesd
# ---------------------------------------------------------------------------


class TestOutliersGesd:
    """Tests for _outliers_gesd: generalised ESD outlier detection."""

    def test_uniform_data_has_no_outliers(self) -> None:
        """Tightly clustered data must produce an all-False mask."""
        # Arrange
        data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])

        # Act
        mask = _outliers_gesd(data, max_outliers=3, alpha=0.05)

        # Assert
        assert not np.any(mask)

    def test_single_extreme_value_is_flagged(self) -> None:
        """One value far from the cluster must be identified as an outlier."""
        # Arrange
        data = np.array([1.0, 1.1, 0.9, 1.05, 100.0])

        # Act
        mask = _outliers_gesd(data, max_outliers=3, alpha=0.05)

        # Assert — only the last element (100.0) should be flagged
        assert mask[-1]
        assert mask.sum() == 1

    def test_two_outliers_both_flagged(self) -> None:
        """Two extreme values at opposite ends must both be identified as outliers."""
        # Arrange
        data = np.array([1.0, 1.1, 0.9, 1.05, 100.0, -100.0])

        # Act
        mask = _outliers_gesd(data, max_outliers=4, alpha=0.05)

        # Assert
        assert mask[4] and mask[5]
        assert mask.sum() == 2


# ---------------------------------------------------------------------------
# _get_esd_criterion
# ---------------------------------------------------------------------------


class TestGetEsdCriterion:
    """Tests for _get_esd_criterion: returns inlier mask via GESD test."""

    def test_inlier_mask_is_complement_of_outlier_mask(self) -> None:
        """The inlier mask must be the boolean complement of the GESD outlier mask."""
        # Arrange
        data = np.array([0.0, 0.1, 0.05, -0.05, 50.0])

        # Act
        inlier_mask = _get_esd_criterion(data)

        # Assert — the single outlier (50.0) must be excluded
        assert not inlier_mask[-1]
        assert inlier_mask[:-1].all()

    def test_uniform_data_all_inliers(self) -> None:
        """Uniform data must yield an all-True inlier mask."""
        # Arrange
        data = np.zeros(6)

        # Act
        inlier_mask = _get_esd_criterion(data)

        # Assert
        assert inlier_mask.all()


# ---------------------------------------------------------------------------
# _get_threshold_criterion
# ---------------------------------------------------------------------------


class TestGetThresholdCriterion:
    """Tests for _get_threshold_criterion: inlier mask based on 2× threshold."""

    def test_values_within_double_threshold_are_inliers(self) -> None:
        """Values with |x| ≤ 2 * threshold must be marked as inliers."""
        # Arrange
        values = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        threshold = 0.6  # 2 * threshold = 1.2

        # Act
        mask = _get_threshold_criterion(values, threshold)

        # Assert — all |values| ≤ 1.2, so all should be inliers
        assert mask.all()

    def test_values_exceeding_double_threshold_are_outliers(self) -> None:
        """Values with |x| > 2 * threshold must be marked as outliers."""
        # Arrange
        values = np.array([0.0, 0.5, 5.0])
        threshold = 1.0  # 2 * threshold = 2.0

        # Act
        mask = _get_threshold_criterion(values, threshold)

        # Assert — 5.0 > 2.0, so the last entry must be False
        assert mask[0] and mask[1]
        assert not mask[2]
