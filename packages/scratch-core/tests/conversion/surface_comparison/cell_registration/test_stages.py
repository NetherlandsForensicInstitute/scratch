"""Tests for conversion.surface_comparison.cell_registration.stages."""

import numpy as np
import pytest
import torch

from conversion.surface_comparison.cell_registration.geometry import (
    pad_image_array,
    rotate_image,
)
from conversion.surface_comparison.cell_registration.search import find_best_matches
from conversion.surface_comparison.cell_registration.stages import run_fine_stage

from .helpers import downsample, make_surface, match_coarse_to_fine

DEVICE = torch.device("cpu")


class TestRunFineStage:
    def test_returns_empty_for_no_templates(self):
        assert (
            run_fine_stage(
                image_full=np.zeros((10, 10)),
                templates_full=[],
                candidates=[],
                coarse_cell_shape=(1, 1),
                coarse_image_shape=(10, 10),
                cap_factor=1.0,
                angles=np.array([0.0]),
                position_margin=1,
                angle_margin_degrees=1.0,
                minimum_fill_fraction=0.9,
                fill_value_full=0.0,
                device=DEVICE,
            )
            == []
        )


class TestMatchCoarseToFine:
    ANGLES = np.arange(-4.0, 4.001, 0.5)
    CELL = 60

    @pytest.fixture
    def case(self):
        """A padded comparison image plus cells cut from a rotated copy of it, at two scales."""
        surface = make_surface(420, 400, seed=8)
        padded_full = pad_image_array(surface, self.CELL, self.CELL)
        rotated = rotate_image(surface, 2.0, fill_value=np.nan).astype(np.float64)
        corners = [(120, 130), (200, 210), (260, 140)]
        templates_full = [
            rotated[top : top + self.CELL, left : left + self.CELL].copy()
            for top, left in corners
        ]
        assert all(np.isfinite(t).all() for t in templates_full)

        cap_factor = 4.0
        coarse_surface = downsample(surface, cap_factor)
        coarse_cell = max(1, round(self.CELL / cap_factor))
        padded_coarse = pad_image_array(coarse_surface, coarse_cell, coarse_cell)
        rotated_coarse = downsample(rotated, cap_factor)
        templates_coarse = [
            rotated_coarse[
                round(top / cap_factor) : round(top / cap_factor) + coarse_cell,
                round(left / cap_factor) : round(left / cap_factor) + coarse_cell,
            ].copy()
            for top, left in corners
        ]
        templates_coarse = [
            np.nan_to_num(t, nan=float(np.nanmean(t))) for t in templates_coarse
        ]

        return {
            "image_full": padded_full,
            "image_coarse": padded_coarse,
            "templates_full": templates_full,
            "templates_coarse": templates_coarse,
            "cap_factor": cap_factor,
            "fill_value": float(np.nanmean(surface)),
        }

    @staticmethod
    def _run(case, angles, **kwargs):
        return match_coarse_to_fine(
            case["image_full"],
            case["image_coarse"],
            case["templates_full"],
            case["templates_coarse"],
            case["cap_factor"],
            angles,
            0.9,
            case["fill_value"],
            case["fill_value"],
            device=DEVICE,
            **kwargs,
        )

    def test_returns_empty_for_no_templates(self):
        assert (
            match_coarse_to_fine(
                np.zeros((10, 10)),
                np.zeros((10, 10)),
                [],
                [],
                1.0,
                self.ANGLES,
                0.9,
                0.0,
                0.0,
            )
            == []
        )

    def test_rejects_templates_of_differing_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            match_coarse_to_fine(
                np.zeros((60, 60)),
                np.zeros((60, 60)),
                [np.zeros((10, 10)), np.zeros((12, 12))],
                [np.zeros((10, 10)), np.zeros((10, 10))],
                1.0,
                self.ANGLES,
                0.9,
                0.0,
                0.0,
            )

    def test_rejects_misaligned_template_lists(self):
        with pytest.raises(ValueError, match="aligned"):
            match_coarse_to_fine(
                np.zeros((60, 60)),
                np.zeros((30, 30)),
                [np.zeros((10, 10))],
                [np.zeros((5, 5)), np.zeros((5, 5))],
                2.0,
                self.ANGLES,
                0.9,
                0.0,
                0.0,
            )

    def test_constant_template_is_rejected(self, case):
        # Arrange: a featureless cell has no defined correlation.
        constant_full = np.full_like(case["templates_full"][0], 5.0)
        constant_coarse = np.full_like(case["templates_coarse"][0], 5.0)

        # Act
        results = match_coarse_to_fine(
            case["image_full"],
            case["image_coarse"],
            [constant_full],
            [constant_coarse],
            case["cap_factor"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            case["fill_value"],
            device=DEVICE,
        )

        # Assert
        assert results[0].score == -1.0

    def test_agrees_with_the_exhaustive_search_when_cap_factor_is_one(self, case):
        # Arrange: cap_factor <= 1 takes the single-pass shortcut straight to find_best_matches.
        exhaustive = find_best_matches(
            case["image_full"],
            case["templates_full"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            device=DEVICE,
        )
        shortcut = match_coarse_to_fine(
            case["image_full"],
            case["image_full"],
            case["templates_full"],
            case["templates_full"],
            1.0,
            self.ANGLES,
            0.9,
            case["fill_value"],
            case["fill_value"],
            device=DEVICE,
        )

        # Assert
        assert shortcut == exhaustive

    def test_agrees_approximately_with_the_exhaustive_search(self, case):
        # Arrange
        exhaustive = find_best_matches(
            case["image_full"],
            case["templates_full"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            device=DEVICE,
        )

        # Act
        coarse = self._run(case, self.ANGLES)

        # Assert
        for reference, other in zip(exhaustive, coarse):
            assert other.angle_deg == pytest.approx(reference.angle_deg)
            assert (other.x, other.y) == (reference.x, reference.y)
            assert other.score == pytest.approx(reference.score, abs=1e-2)

    def test_recovers_the_planted_rotation(self, case):
        # Act
        results = self._run(case, self.ANGLES)

        # Assert
        for match in results:
            assert match.angle_deg == pytest.approx(2.0)
            assert match.score > 0.95

    def test_score_stays_within_the_valid_pearson_range(self, case):
        # Act
        results = self._run(case, self.ANGLES)

        # Assert: the rejection sentinel must never leak out as a reported score.
        for match in results:
            assert -1.0 <= match.score <= 1.0

    def test_more_candidates_never_lowers_the_score(self, case):
        # Arrange: extra candidates add search, so a score can only improve or stay equal.
        few = self._run(case, self.ANGLES, n_candidates=1)
        many = self._run(case, self.ANGLES, n_candidates=4)

        # Assert
        for lower, higher in zip(few, many):
            assert higher.score >= lower.score - 1e-6
