"""
Unit tests for the private helper functions in cmc_classification.
"""

import numpy as np
import pytest

from conversion.surface_comparison.cmc_classification import (
    _circular_median,
    _get_consensus_angle,
    _get_consensus_translation,
    _get_esd_criterion,
    _get_threshold_criterion,
    _outliers_gesd,
    _rosner_critical_value,
    rotate_points,
    _wrap_angles,
)
from conversion.surface_comparison.models import Cell, CellMetaData
from ..helper_functions import make_cell


def test_cell_size_um_converts_meters_to_micrometers():
    cell = Cell(
        center_reference=(0.0, 0.0),
        cell_size=(50e-6, 100e-6),
        fill_fraction_reference=1.0,
        best_score=0.5,
        angle_deg=0.0,
        center_comparison=(0.0, 0.0),
        is_congruent=False,
        meta_data=CellMetaData(
            is_outlier=False, residual_angle_deg=0, position_error=(0, 0)
        ),
    )

    width_um, height_um = cell.cell_size_um

    assert width_um == pytest.approx(50.0)
    assert height_um == pytest.approx(100.0)


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


class TestCircularMedian:
    """Tests for _circular_median: computes the circular median of radian angles."""

    def test_tight_cluster_returns_cluster_centre(self) -> None:
        """A tight cluster of similar angles should return a value near their mean."""
        # Arrange
        angles = np.radians(np.array([10.0, 11.0, 10.5, 9.8, 10.2]))

        # Act
        result = _circular_median(angles)

        # Assert — result must be within 1° of the cluster center
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


class TestRotatePoints:
    """Tests for _rotate_points: rotates (N, 2) point arrays around a center."""

    def test_zero_rotation_returns_original_points(self) -> None:
        """Rotating by zero radians must leave every point unchanged."""
        # Arrange
        points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        center = 0.0, 0.0

        # Act
        result = rotate_points(points, angle=0.0, center=center)

        # Assert
        np.testing.assert_allclose(result, points, atol=1e-12)

    def test_quarter_turn_around_origin(self) -> None:
        """A 90° CCW rotation of [1, 0] around the origin should give [0, 1]."""
        # Arrange
        points = np.array([[1.0, 0.0]])
        center = 0.0, 0.0

        # Act
        result = rotate_points(points, angle=np.pi / 2, center=center)

        # Assert
        np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-12)

    def test_rotation_around_non_origin_center(self) -> None:
        """Rotating a point 180° around itself must return the same point."""
        # Arrange
        points = np.array([[3.0, 4.0]])
        center = 3.0, 4.0

        # Act
        result = rotate_points(points, angle=np.pi, center=center)

        # Assert
        np.testing.assert_allclose(result, points, atol=1e-12)


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


class TestOutliersGESD:
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


class TestGetConsensusAngle:
    """Tests for _get_consensus_angle: three-step median-and-rejection procedure."""

    def test_uniform_angles_returns_their_common_value(self) -> None:
        """When all cells share the same angle the consensus must equal that angle."""
        # Arrange
        angle_deg = 15.0
        cells = [make_cell(angle_deg=angle_deg) for _ in range(6)]
        threshold = np.radians(2.0)

        # Act
        result = _get_consensus_angle(cells=cells, threshold=threshold)

        # Assert
        np.testing.assert_allclose(np.degrees(result), angle_deg, atol=1e-6)

    def test_single_extreme_outlier_does_not_shift_consensus(self) -> None:
        """One extreme angle must be rejected so the consensus stays near the cluster."""
        # Arrange
        inlier_angle = 10.0
        cells = [make_cell(angle_deg=inlier_angle) for _ in range(7)]
        cells.append(make_cell(angle_deg=170.0))  # extreme outlier
        threshold = np.radians(2.0)

        # Act
        result = _get_consensus_angle(cells=cells, threshold=threshold)

        # Assert — consensus must be near the inlier cluster, not pulled toward 170°
        assert abs(np.degrees(result) - inlier_angle) < 1.0

    def test_outlier_cells_are_flagged_in_meta_data(self) -> None:
        """Cells outside the acceptance band must have is_outlier set to True."""
        # Arrange
        cells = [make_cell(angle_deg=5.0) for _ in range(6)]
        cells.append(make_cell(angle_deg=170.0))  # will be rejected
        threshold = np.radians(2.0)

        # Act
        _get_consensus_angle(cells=cells, threshold=threshold)

        # Assert — the extreme cell must be marked as an outlier
        assert cells[-1].meta_data.is_outlier
        assert all(not c.meta_data.is_outlier for c in cells[:-1])

    def test_residual_angle_deg_populated_for_all_cells(self) -> None:
        """residual_angle_deg must be set on every cell after the call."""
        # Arrange
        cells = [make_cell(angle_deg=float(a)) for a in [1.0, 2.0, 1.5, 1.8, 1.2]]
        threshold = np.radians(2.0)

        # Act
        _get_consensus_angle(cells=cells, threshold=threshold)

        # Assert — every residual must be a finite float
        for cell in cells:
            assert np.isfinite(cell.meta_data.residual_angle_deg)


class TestGetConsensusTranslation:
    """Tests for _get_consensus_translation: median offset after rotating reference centers."""

    def test_zero_angle_and_zero_offset_gives_zero_translation(self) -> None:
        """When reference and comparison centers coincide the consensus translation must be zero."""
        # Arrange
        cells = [
            make_cell(
                angle_deg=0.0,
                center_reference=(float(x), float(y)),
                center_comparison=(float(x), float(y)),
                is_outlier=False,
            )
            for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        ]
        rotation_center = (0.0, 0.0)

        # Act
        tx, ty = _get_consensus_translation(
            cells=cells, angle=0.0, rotation_center=rotation_center
        )

        # Assert
        np.testing.assert_allclose([tx, ty], [0.0, 0.0], atol=1e-12)

    def test_uniform_offset_is_recovered_as_consensus(self) -> None:
        """A constant displacement applied to every comparison center must equal the consensus."""
        # Arrange
        offset = (0.5, -0.3)
        centers = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        cells = [
            make_cell(
                angle_deg=0.0,
                center_reference=c,
                center_comparison=(c[0] + offset[0], c[1] + offset[1]),
                is_outlier=False,
            )
            for c in centers
        ]
        rotation_center = (0.0, 0.0)

        # Act
        tx, ty = _get_consensus_translation(
            cells=cells, angle=0.0, rotation_center=rotation_center
        )

        # Assert
        np.testing.assert_allclose([tx, ty], list(offset), atol=1e-10)

    def test_outlier_cells_excluded_from_translation(self) -> None:
        """Cells flagged as outliers must not influence the consensus translation."""
        # Arrange
        good_offset = (0.2, 0.1)
        centers = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        cells = [
            make_cell(
                angle_deg=0.0,
                center_reference=c,
                center_comparison=(c[0] + good_offset[0], c[1] + good_offset[1]),
                is_outlier=False,
            )
            for c in centers
        ]
        # Add an outlier cell with a wildly different offset
        outlier = make_cell(
            angle_deg=0.0,
            center_reference=(2.0, 2.0),
            center_comparison=(2.0 + 999.0, 2.0 + 999.0),
            is_outlier=True,
        )
        cells.append(outlier)
        rotation_center = (0.0, 0.0)

        # Act
        tx, ty = _get_consensus_translation(
            cells=cells, angle=0.0, rotation_center=rotation_center
        )

        # Assert — result must be near the inlier offset, not dragged toward 999
        np.testing.assert_allclose([tx, ty], list(good_offset), atol=1e-10)

    def test_position_error_set_on_all_cells(self) -> None:
        """position_error must be populated for every cell after the call."""
        # Arrange
        cells = [
            make_cell(
                angle_deg=0.0,
                center_reference=(float(i), 0.0),
                center_comparison=(float(i) + 0.1, 0.0),
                is_outlier=False,
            )
            for i in range(5)
        ]
        rotation_center = (0.0, 0.0)

        # Act
        _get_consensus_translation(
            cells=cells, angle=0.0, rotation_center=rotation_center
        )

        # Assert
        for cell in cells:
            assert len(cell.meta_data.position_error) == 2
            assert all(np.isfinite(e) for e in cell.meta_data.position_error)
