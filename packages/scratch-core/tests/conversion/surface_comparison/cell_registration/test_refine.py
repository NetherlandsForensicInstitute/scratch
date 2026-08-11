"""Tests for conversion.surface_comparison.cell_registration.refine."""

import numpy as np
import pytest
import torch

from conversion.surface_comparison.cell_registration.geometry import (
    map_image_to_canvas,
    pad_image_array,
)
from conversion.surface_comparison.cell_registration.models import RefinementJob
from conversion.surface_comparison.cell_registration.refine import refine
from conversion.surface_comparison.cell_registration.scoring import (
    compute_mean_and_std,
    prepare_templates,
)

from .helpers import make_surface

DEVICE = torch.device("cpu")


class TestRefine:
    CELL = 40
    MARGIN = 8

    @pytest.fixture
    def case(self):
        """A padded image, one cell cut from a known location, and its true center."""
        surface = make_surface(200, 180, seed=3)
        padded = pad_image_array(surface, self.CELL, self.CELL)
        top, left = 60 + self.CELL, 70 + self.CELL  # position within the padded image
        template = padded[top : top + self.CELL, left : left + self.CELL].copy()
        center = (left + self.CELL / 2, top + self.CELL / 2)
        return padded, template, center

    @staticmethod
    def _run(padded, templates, jobs, margin, batch_size=256):
        tensor, _ = prepare_templates(templates, DEVICE)
        return refine(
            padded.astype(np.float32),
            tensor,
            jobs,
            margin,
            0.9,
            float(np.nanmean(padded)),
            compute_mean_and_std(padded),
            batch_size,
            0.0,
        )

    def test_recovers_a_planted_match(self, case):
        # Arrange
        padded, template, center = case
        jobs = [RefinementJob(0, center[0], center[1], 0.0)]

        # Act
        match = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert match.score == pytest.approx(1.0, abs=1e-4)
        assert match.angle_deg == 0.0
        expected_left, expected_top = map_image_to_canvas(
            center[0], center[1], (self.CELL, self.CELL), padded.shape, 0.0
        )
        assert (match.x, match.y) == (round(expected_left), round(expected_top))

    def test_finds_a_match_offset_within_the_margin(self, case):
        # Arrange: predicted center is wrong by less than the search radius.
        padded, template, center = case
        jobs = [
            RefinementJob(
                0, center[0] + self.MARGIN - 2, center[1] - self.MARGIN + 2, 0.0
            )
        ]

        # Act
        match = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert match.score == pytest.approx(1.0, abs=1e-4)
        expected_left, expected_top = map_image_to_canvas(
            center[0], center[1], (self.CELL, self.CELL), padded.shape, 0.0
        )
        assert (match.x, match.y) == (round(expected_left), round(expected_top))

    def test_misses_a_match_outside_the_margin(self, case):
        # Arrange: predicted center is wrong by far more than the search radius.
        padded, template, center = case
        jobs = [RefinementJob(0, center[0] + 6 * self.MARGIN, center[1], 0.0)]

        # Act
        match = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert match.score < 0.99

    def test_keeps_the_best_pose_across_several_jobs(self, case):
        # Arrange: the true pose competes with two decoys for the same cell.
        padded, template, center = case
        jobs = [
            RefinementJob(0, center[0] + 6 * self.MARGIN, center[1], 0.0),
            RefinementJob(0, center[0], center[1], 0.0),
            RefinementJob(0, center[0], center[1] + 6 * self.MARGIN, 0.0),
        ]

        # Act
        match = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert match.score == pytest.approx(1.0, abs=1e-4)

    def test_chunking_does_not_change_the_result(self, case):
        # Arrange: decoy positions at a single angle, so one pose is the clear winner.
        padded, template, center = case
        jobs = [
            RefinementJob(0, center[0] + offset, center[1], 0.0)
            for offset in (-6 * self.MARGIN, 0.0, 6 * self.MARGIN)
        ]
        single = self._run(padded, [template], jobs, self.MARGIN, batch_size=100)
        split = self._run(padded, [template], jobs, self.MARGIN, batch_size=1)

        # Assert: the pose must be identical; the score differs by float32 FFT noise, since chunk
        # size changes the batch handed to the transform.
        assert [match[1:] for match in single] == [match[1:] for match in split]
        for one, other in zip(single, split):
            assert one.score == pytest.approx(other.score, abs=1e-6)

    def test_chunking_agrees_on_score_across_tied_angles(self, case):
        padded, template, center = case
        jobs = [
            RefinementJob(0, center[0], center[1], float(a))
            for a in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        single = self._run(padded, [template], jobs, self.MARGIN, batch_size=100)
        split = self._run(padded, [template], jobs, self.MARGIN, batch_size=2)
        assert single[0].score == pytest.approx(split[0].score, abs=1e-5)

    def test_cell_without_jobs_keeps_the_default(self, case):
        # Arrange: two cells, but only the first is given a job.
        padded, template, center = case
        jobs = [RefinementJob(0, center[0], center[1], 0.0)]

        # Act
        results = self._run(padded, [template, template.copy()], jobs, self.MARGIN)

        # Assert
        assert results[1] == (-np.inf, 0, 0, 0.0)
