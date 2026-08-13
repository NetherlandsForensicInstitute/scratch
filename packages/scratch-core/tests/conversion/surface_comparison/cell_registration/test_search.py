"""Tests for conversion.surface_comparison.cell_registration.search."""

import numpy as np
import pytest
import torch

from conversion.surface_comparison.cell_registration.geometry import (
    pad_image_array,
    rotate_image,
)
from conversion.surface_comparison.cell_registration.search import (
    search_candidates,
    sort_by_absolute_angle,
)

from .helpers import make_surface

DEVICE = torch.device("cpu")


class TestSortByAbsoluteAngle:
    def test_orders_by_magnitude_then_sign(self):
        result = sort_by_absolute_angle(np.array([5.0, -1.0, 0.0, 1.0, -5.0]))
        assert list(result) == [0.0, -1.0, 1.0, -5.0, 5.0]


class TestSearchCandidates:
    # search_candidates operates end-to-end on an image + templates rather than a pre-built score
    # volume, so these are expressed as small planted-match cases instead of directly driving the
    # internal peak-extraction step.

    @pytest.fixture
    def multi_peak_case(self):
        """A surface with three well-separated, equally strong patches cut as one template."""
        surface = make_surface(80, 80, seed=1)
        template = surface[10:20, 10:20].copy()
        canvas = np.full((120, 120), float(np.nanmean(surface)))
        for top, left in [(10, 10), (60, 60), (10, 90)]:
            canvas[top : top + 10, left : left + 10] = template
        return canvas, template

    def test_returns_empty_for_no_templates(self):
        assert (
            search_candidates(
                np.zeros((10, 10)), [], np.array([0.0]), 0.9, 0.0, device=DEVICE
            )
            == []
        )

    def test_rejects_templates_of_differing_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            search_candidates(
                np.zeros((60, 60)),
                [np.zeros((10, 10)), np.zeros((12, 12))],
                np.array([0.0]),
                0.9,
                0.0,
                device=DEVICE,
            )

    def test_returns_requested_number_of_candidates(self, multi_peak_case):
        canvas, template = multi_peak_case
        candidates = search_candidates(
            canvas,
            [template],
            np.array([0.0]),
            0.9,
            float(np.nanmean(canvas)),
            n_candidates=3,
            suppression_radius=3,
            device=DEVICE,
        )
        assert len(candidates[0]) == 3

    def test_orders_candidates_by_score(self, multi_peak_case):
        canvas, template = multi_peak_case
        candidates = search_candidates(
            canvas,
            [template],
            np.array([0.0]),
            0.9,
            float(np.nanmean(canvas)),
            n_candidates=3,
            suppression_radius=3,
            device=DEVICE,
        )
        scores = [match.score for match in candidates[0]]
        assert scores == sorted(scores, reverse=True)

    def test_reports_the_best_angle_at_each_location(self):
        surface = make_surface(100, 100, seed=4)
        rotated = rotate_image(surface, 5.0, fill_value=np.nan).astype(np.float64)
        template = rotated[40:70, 40:70].copy()
        assert np.isfinite(template).all()
        padded = pad_image_array(surface, 30, 30)

        angles = np.union1d(np.arange(-8.0, 8.001, 1.0), [5.0])
        candidates = search_candidates(
            padded,
            [template],
            angles,
            0.9,
            float(np.nanmean(surface)),
            n_candidates=1,
            device=DEVICE,
        )
        assert candidates[0][0].angle_deg == pytest.approx(5.0, abs=1.0)
        assert candidates[0][0].score > 0.9

    def test_suppresses_neighbors_of_a_found_peak(self, multi_peak_case):
        canvas, template = multi_peak_case
        candidates = search_candidates(
            canvas,
            [template],
            np.array([0.0]),
            0.9,
            float(np.nanmean(canvas)),
            n_candidates=3,
            suppression_radius=15,
            device=DEVICE,
        )
        positions = [(match.x, match.y) for match in candidates[0]]
        # With a large suppression radius, candidates must be well separated from each other.
        for i, (x0, y0) in enumerate(positions):
            for x1, y1 in positions[i + 1 :]:
                assert max(abs(x0 - x1), abs(y0 - y1)) >= 15

    def test_perfect_anti_correlation_is_not_mistaken_for_rejection(self):
        # An empty list is the only signal for "no candidate", and this search decides where each
        # cell gets refined, so a cell discarded here loses its match entirely rather than merely
        # scoring badly.
        # Arrange: Match template and canvas dimensions so there is only 1 candidate position.
        surface = make_surface(20, 20, seed=2)
        template = surface.copy()

        # Invert canvas for perfect anti-correlation (-1.0)
        canvas = 2 * np.nanmean(template) - template

        # Zero padding ensures argmax cannot choose a misaligned position with higher score
        padded = pad_image_array(canvas, 0, 0)

        # Act
        candidates = search_candidates(
            padded,
            [template],
            np.array([0.0]),
            0.9,
            float(np.nanmean(padded)),
            n_candidates=1,
            device=DEVICE,
        )

        # Assert
        assert candidates[0]
        assert candidates[0][0].score == pytest.approx(-1.0, abs=1e-3)

    def test_handles_each_template_independently(self):
        surface = make_surface(120, 120, seed=6)
        padded = pad_image_array(surface, 20, 20)
        template_a = padded[30:50, 30:50].copy()
        template_b = padded[80:100, 60:80].copy()

        candidates = search_candidates(
            padded,
            [template_a, template_b],
            np.array([0.0]),
            0.9,
            float(np.nanmean(surface)),
            n_candidates=1,
            device=DEVICE,
        )
        assert len(candidates) == 2
        assert all(candidates)

    def test_constant_template_yields_no_candidate(self):
        surface = make_surface(60, 60, seed=7)
        padded = pad_image_array(surface, 15, 15)
        constant = np.full((15, 15), 5.0)

        candidates = search_candidates(
            padded,
            [constant],
            np.array([0.0]),
            0.9,
            float(np.nanmean(surface)),
            n_candidates=3,
            device=DEVICE,
        )
        assert candidates[0] == []
