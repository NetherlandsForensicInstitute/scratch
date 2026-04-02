import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from conversion.surface_comparison.cmc_consensus.procrustes import (
    find_consensus_parameters,
    _get_rotation_component_using_rotation_matrix,
    _get_rotation_component_using_angle_degree,
    _build_2d_rotation_matrix,
)

from conversion.surface_comparison.cmc_consensus.criterion import (
    calculate_criterion,
    _get_cell_angle_and_position_distances,
    _get_distances,
    _predict_positions,
    _get_distances_meters,
)

from conversion.surface_comparison.models import Cell

from conversion.surface_comparison.cmc_consensus.models import ConsensusParameters


class TestCriterion:
    def test_empty_cell_ids_returns_inf(self):
        """Test that empty cell_ids returns np.inf."""
        cell_distances = np.array([1.0, 2.0, 3.0])
        cell_angle_distances = np.array([10.0, 20.0, 30.0])
        result = calculate_criterion(
            [], cell_distances, cell_angle_distances, 10.0, 90.0
        )
        assert result == np.inf

    def test_basic_calculation(self):
        """Test criterion calculation with known values."""
        cell_ids = [0, 1]
        cell_distances = np.array([2.0, 4.0])  # mean = 3.0
        cell_angle_distances = np.array([10.0, 30.0])  # mean = 20.0
        max_distance = 10.0
        max_abs_angle_distance = 40.0

        result = calculate_criterion(
            cell_ids,
            cell_distances,
            cell_angle_distances,
            max_distance,
            max_abs_angle_distance,
        )

        expected = 3.0 / 10.0 + 20.0 / 40.0  # 0.3 + 0.5 = 0.8
        assert result == pytest.approx(expected)


class TestGetCmcConsensus:
    def test_uses_only_included_idx_for_least_squares(self):
        """Test that find_consensus_parameters is called with only the included cells, not all cells."""
        all_cells = [MagicMock(spec=Cell) for _ in range(5)] # type: ignore[list-item] 
        included_idx = [0, 2, 4]

        mock_consensus_params = ConsensusParameters(
            rotation_center_reference=np.array([0.0, 0.0]),
            rotation_center_comparison=np.array([1.0, 1.0]),
            rotation_rad=0.1,
        )

        with (
            patch(
                "conversion.surface_comparison.cmc_consensus.criterion.find_consensus_parameters",
                return_value=mock_consensus_params,
            ) as mock_find,
            patch(
                "conversion.surface_comparison.cmc_consensus.criterion._get_distances",
                return_value=(np.zeros(5), np.zeros(5)),
            ),
        ):
            _get_cell_angle_and_position_distances(included_idx, all_cells)

            mock_find.assert_called_once_with(
                [all_cells[0], all_cells[2], all_cells[4]]
            )


class TestFindConsensusParameters:
    @pytest.mark.parametrize(
        "angle, translation",
        [
            (np.pi / 6, [3.0, 5.0]),
            (np.pi / 4, [0.0, 0.0]),
            (-np.pi / 3, [-2.0, 7.0]),
            (0.0, [1.5, -3.5]),
            (np.pi / 2, [10.0, 0.0]),
        ],
    )
    def test_recovers_all_parameters(self, angle, translation):
        """Test that all outputs match known rotation, translation, and centers."""
        translation = np.array(translation)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

        centers_reference = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        centers_comparison = centers_reference @ rotation_matrix + translation

        cells = [
            MagicMock(
                spec=Cell,
                center_reference=ref.tolist(),
                center_comparison=comp.tolist(),
            )
            for ref, comp in zip(centers_reference, centers_comparison)
        ]

        result = find_consensus_parameters(cells)

        expected_rotation_center_reference = centers_reference.mean(axis=0)
        expected_rotation_center_comparison = centers_comparison.mean(axis=0)
        expected_translation = (
            expected_rotation_center_comparison - expected_rotation_center_reference
        )

        assert result.rotation_rad == pytest.approx(angle, abs=1e-6)
        assert result.rotation_center_reference == pytest.approx(
            expected_rotation_center_reference, abs=1e-6
        )
        assert result.rotation_center_comparison == pytest.approx(
            expected_rotation_center_comparison, abs=1e-6
        )
        # translation is derived from the two centers, so verify it is consistent
        actual_translation = (
            result.rotation_center_comparison - result.rotation_center_reference
        )
        assert actual_translation == pytest.approx(expected_translation, abs=1e-6)


class TestRotationComponentWithRotationMatrix:
    @pytest.mark.parametrize("angle", np.linspace(-np.pi, np.pi, 13))
    def test_known_rotation_returns_correct_component(self, angle):
        """Test that rotation around a center returns (data - center) rotated by the given angle."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

        data = np.array([[2.0, 1.0], [0.0, 3.0]])
        center = np.array([[1.0, 1.0]])

        result = _get_rotation_component_using_rotation_matrix(
            data, center, rotation_matrix
        )

        expected = (data - center) @ rotation_matrix
        assert result == pytest.approx(expected, abs=1e-6)


class TestGetDistances:
    @pytest.mark.parametrize("consensus_rotation_rad", [np.pi / 6, -np.pi / 3])
    @pytest.mark.parametrize("rotation_center_reference", [(0.0, 0.0), (3.0, -2.0)])
    @pytest.mark.parametrize("rotation_center_comparison", [(0.0, 0.0), (1.0, 5.0)])
    def test_returns_distances_and_angle_distances_for_known_inputs(
        self,
        consensus_rotation_rad,
        rotation_center_reference,
        rotation_center_comparison,
    ):
        """Test that distances and abs_angle_distances are correctly computed for known inputs."""
        consensus_rotation_deg = np.degrees(consensus_rotation_rad)

        cell_angle_degs = [10.0, -50.0, 0.0]
        cells = [MagicMock(spec=Cell, angle_deg=a) for a in cell_angle_degs]

        expected_distances = np.array([1.0, 2.0, 3.0])
        # The absolute residual is |cell.angle_deg - -consensus_rotation_deg|, since we use pixel_coordinates for rotation_angle of cells and mathematical coordinates here.
        expected_abs_angle_distances = np.array(
            [abs(a - -consensus_rotation_deg) for a in cell_angle_degs]
        )

        with (
            patch(
                "conversion.surface_comparison.cmc_consensus.criterion._predict_positions",
                return_value=MagicMock(),
            ),
            patch(
                "conversion.surface_comparison.cmc_consensus.criterion._get_distances_meters",
                return_value=expected_distances.tolist(),
            ),
        ):
            distances, abs_angle_distances = _get_distances(
                cells,
                consensus_rotation_rad,
                rotation_center_reference,
                rotation_center_comparison,
            )

        assert distances == pytest.approx(expected_distances, abs=1e-6)
        assert abs_angle_distances == pytest.approx(
            expected_abs_angle_distances, abs=1e-6
        )


class TestRotateUsingAngleDeg:
    @pytest.mark.parametrize("angle_deg", np.linspace(-180, 180, 13))
    def test_rotation_recovers_known_output(self, angle_deg):
        """Test that rotating data around a center returns the correct rotated positions."""
        xy_data = np.array([[2.0, 1.0], [0.0, 3.0]])
        reference_center = np.array([1.0, 1.0])

        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        expected = (xy_data - reference_center.reshape(1, -1)) @ rotation_matrix

        result = _get_rotation_component_using_angle_degree(
            xy_data, angle_deg, reference_center
        )

        assert result == pytest.approx(expected, abs=1e-6)


class TestBuild2dRotationMatrix:
    @pytest.mark.parametrize("angle_rad", np.linspace(-np.pi, np.pi, 13))
    def test_rotation_matrix_has_correct_values(self, angle_rad):
        """Test that the rotation matrix has correct cos/sin entries and is orthonormal."""
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        expected = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

        result = _build_2d_rotation_matrix(angle_rad)

        assert result == pytest.approx(expected, abs=1e-6)
        assert result.shape == (2, 2)

        # assert two properties of rotation matrix: determinant of 1 and 'rotate -> undo rotate' gives eye.
        assert np.linalg.det(result) == pytest.approx(1.0, abs=1e-6)
        assert result @ result.T == pytest.approx(np.eye(2), abs=1e-6)


class TestPredictPositions:
    @pytest.mark.parametrize("consensus_rotation_rad", [np.pi / 6, -np.pi / 3])
    @pytest.mark.parametrize(
        "rotation_center_reference", [np.array([0.0, 0.0]), np.array([3.0, -2.0])]
    )
    @pytest.mark.parametrize("rotation_center_comparison", [(0.0, 0.0), (1.0, 5.0)])
    def test_predicts_correct_positions(
        self,
        consensus_rotation_rad,
        rotation_center_reference,
        rotation_center_comparison,
    ):
        """Test that predicted positions match manual rotation + translation computation."""
        centers_reference = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        cells = [
            MagicMock(spec=Cell, center_reference=c.tolist()) for c in centers_reference
        ]

        cos_a, sin_a = np.cos(consensus_rotation_rad), np.sin(consensus_rotation_rad)
        rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        rotation_center = np.array(rotation_center_reference).reshape(1, 2)
        expected = (
            centers_reference - rotation_center
        ) @ rotation_matrix + rotation_center_comparison

        result = _predict_positions(
            cells,
            consensus_rotation_rad,
            rotation_center_reference,
            rotation_center_comparison,
        )

        assert result == pytest.approx(expected, abs=1e-6)


class TestGetDistancesMeters:
    @pytest.mark.parametrize(
        "centers_comparison, predicted_positions",
        [
            ([[0.0, 0.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]),  # zero and 5.0
            ([[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]),  # all zeros
            ([[4.0, 0.0], [0.0, 3.0]], [[1.0, 0.0], [0.0, 0.0]]),  # 3.0 and 3.0
        ],
    )
    def test_returns_correct_euclidean_distances(
        self, centers_comparison, predicted_positions
    ):
        """Test that Euclidean distances between comparison centers and predicted positions are correct."""
        cells = [MagicMock(spec=Cell, center_comparison=c) for c in centers_comparison]
        predicted_positions = np.array(predicted_positions)

        expected = [
            float(np.sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2))
            for c, p in zip(centers_comparison, predicted_positions)
        ]

        result = _get_distances_meters(cells, predicted_positions)

        assert result == pytest.approx(expected, abs=1e-6)
