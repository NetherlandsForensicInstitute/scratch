"""Tests for conversion.surface_comparison.cell_registration.stages."""

import numpy as np
import pytest
import torch

from conversion.surface_comparison.cell_registration.geometry import (
    pad_image_array,
    rotate_image,
)
from conversion.surface_comparison.cell_registration.search import (
    find_best_matches,
    get_uniform_cell_shape,
)
from conversion.surface_comparison.cell_registration.stages import (
    run_coarse_stage,
    run_fine_stage,
)

from .helpers import downsample, make_surface

DEVICE = torch.device("cpu")


class TestRunCoarseStage:
    def test_returns_empty_for_no_templates(self):
        assert (
            run_coarse_stage(
                image_coarse=np.zeros((10, 10)),
                templates_coarse=[],
                angles=np.array([0.0]),
                minimum_fill_fraction=0.9,
                fill_value_coarse=0.0,
                device=DEVICE,
            )
            == []
        )

    def test_keeps_a_candidate_whose_best_score_is_negative(self):
        # Arrange: a template that is the exact negative of a patch of the image correlates at
        # about -1 there. That is a real, usable candidate, and an empty list is the only signal
        # for "no candidate" - the coarse stage decides where each cell gets refined, so a cell
        # discarded here loses its match entirely rather than merely scoring badly.
        # Sizing the canvas to the template leaves exactly one candidate position, so the best
        # score is the anti-correlated one rather than a positive score somewhere off-alignment.
        template = make_surface(20, 20, seed=3)
        canvas = 2 * np.nanmean(template) - template
        image = pad_image_array(canvas, pad_width=0, pad_height=0)

        # Act
        candidates = run_coarse_stage(
            image_coarse=image,
            templates_coarse=[template],
            angles=np.array([0.0]),
            minimum_fill_fraction=0.9,
            fill_value_coarse=float(np.nanmean(image)),
            n_candidates=1,
            device=DEVICE,
        )

        # Assert
        assert len(candidates[0]) == 1
        assert candidates[0][0].score == pytest.approx(-1.0, abs=1e-3)


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

    def test_rejects_candidates_misaligned_with_the_templates(self):
        # Arrange: candidates are indexed by template, so a length mismatch is a caller error
        # rather than something to silently truncate.
        with pytest.raises(ValueError, match="aligned"):
            run_fine_stage(
                image_full=np.zeros((60, 60)),
                templates_full=[np.zeros((10, 10))],
                candidates=[[], []],
                coarse_cell_shape=(5, 5),
                coarse_image_shape=(30, 30),
                cap_factor=2.0,
                angles=np.array([0.0]),
                position_margin=1,
                angle_margin_degrees=1.0,
                minimum_fill_fraction=0.9,
                fill_value_full=0.0,
                device=DEVICE,
            )


class TestCoarseStageThenFineStage:
    """
    The two runners composed the way the pipeline composes them.

    This is the coarse-to-fine search of the literature, but no function carries that name any
    more: it is :func:`run_coarse_stage` feeding :func:`run_fine_stage`, wired up in
    :func:`~conversion.surface_comparison.pipeline.compare_surfaces`.
    """

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
    def _run(case, angles, templates_full=None, templates_coarse=None, **kwargs):
        """Run the coarse sweep and feed its candidates to the fine refinement."""
        if templates_full is None:
            templates_full = case["templates_full"]
        if templates_coarse is None:
            templates_coarse = case["templates_coarse"]

        candidates = run_coarse_stage(
            image_coarse=case["image_coarse"],
            templates_coarse=templates_coarse,
            angles=angles,
            minimum_fill_fraction=0.9,
            fill_value_coarse=case["fill_value"],
            device=DEVICE,
            **kwargs,
        )
        return run_fine_stage(
            image_full=case["image_full"],
            templates_full=templates_full,
            candidates=candidates,
            coarse_cell_shape=get_uniform_cell_shape(templates_coarse),
            coarse_image_shape=case["image_coarse"].shape,
            cap_factor=case["cap_factor"],
            angles=angles,
            position_margin=5,
            angle_margin_degrees=5.0,
            minimum_fill_fraction=0.9,
            fill_value_full=case["fill_value"],
            device=DEVICE,
        )

    def test_constant_template_is_rejected(self, case):
        # Arrange: a featureless cell has no defined correlation.
        constant_full = np.full_like(case["templates_full"][0], 5.0)
        constant_coarse = np.full_like(case["templates_coarse"][0], 5.0)

        # Act
        results = self._run(
            case,
            self.ANGLES,
            templates_full=[constant_full],
            templates_coarse=[constant_coarse],
        )

        # Assert
        assert results[0].score == -1.0

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
