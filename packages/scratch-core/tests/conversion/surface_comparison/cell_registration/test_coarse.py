import numpy as np
import pytest
import torch

from conversion.surface_comparison.cell_registration.coarse import (
    _to_full_resolution,
    _refine,
    coarse_to_fine_match,
)
from conversion.surface_comparison.cell_registration.utils import (
    canvas_to_image,
    image_to_canvas,
    rotated_shape,
    pad_image_array,
    _prepare_templates,
    batched_match,
    search_candidates,
    REJECTED_SCORE,
    rotate_image,
)
from conversion.resample import resize_nan_aware
from .helpers import (
    make_surface,
)

DEVICE = torch.device("cpu")
PIXEL_SIZE = 1e-6
CELL_TOP_LEFT = (40, 30)
FILL_FRACTION_THRESHOLD = 0.5


def downsample(image, factor):
    """Test-local helper: NaN-aware area-average shrink, matching the old coarse.downsample."""
    height, width = image.shape
    new_shape = (int(np.ceil(height / factor)), int(np.ceil(width / factor)))
    return resize_nan_aware(image, new_shape, interpolation="area")


class TestDownsample:
    def test_output_shape_rounds_up(self):
        assert downsample(np.zeros((10, 10)), 4).shape == (3, 3)
        assert downsample(np.zeros((12, 8)), 4).shape == (3, 2)

    def test_computes_block_mean(self):
        # Arrange: each 2x2 block holds a single distinct value.
        image = np.repeat(np.repeat(np.array([[1.0, 2.0], [3.0, 4.0]]), 2, 0), 2, 1)

        # Act
        result = downsample(image, 2)

        # Assert
        assert result == pytest.approx(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_averages_only_valid_pixels(self):
        # Arrange: one block is half missing; its mean must come from the survivors alone.
        image = np.array([[4.0, 4.0, 1.0, 1.0], [4.0, 4.0, np.nan, np.nan]])

        # Act
        result = downsample(image, 2)

        # Assert
        assert result[0, 0] == pytest.approx(4.0)
        assert result[0, 1] == pytest.approx(1.0)

    def test_block_below_validity_threshold_becomes_nan(self):
        # Arrange: three of four sub-pixels missing, i.e. 25% valid.
        image = np.array([[7.0, np.nan], [np.nan, np.nan]])

        # Act
        result = downsample(image, 2)

        # Assert
        assert np.isnan(result[0, 0])

    def test_fully_missing_block_becomes_nan_without_warning(self):
        # Arrange
        image = np.full((2, 2), np.nan)

        # Act
        with np.errstate(all="raise"):
            result = downsample(image, 2)

        # Assert
        assert np.isnan(result).all()


class TestToFullResolution:
    def test_factor_one_is_identity(self):
        assert _to_full_resolution(7.0, 1) == pytest.approx(7.0)

    @pytest.mark.parametrize(
        ("coarse", "factor", "expected"),
        [(0.0, 4, 1.5), (1.0, 4, 5.5), (0.0, 6, 2.5), (3.0, 2, 6.5)],
    )
    def test_maps_to_block_centre(self, coarse, factor, expected):
        assert _to_full_resolution(coarse, factor) == pytest.approx(expected)

    def test_agrees_with_block_averaging(self):
        # Arrange: coarse pixel i covers full pixels [i*f, (i+1)*f), whose centre is the target.
        factor, index = 5, 3

        # Act
        result = _to_full_resolution(float(index), factor)

        # Assert
        covered = np.arange(index * factor, (index + 1) * factor)
        assert result == pytest.approx(covered.mean())

    def test_agrees_with_block_averaging_for_a_float_factor(self):
        # Arrange: cap_factor is a float in the new pipeline, not necessarily an integer.
        factor, index = 4.5, 2

        # Act
        result = _to_full_resolution(float(index), factor)

        # Assert
        assert result == pytest.approx(index * factor + (factor - 1) / 2.0)


class TestSearchCandidates:
    @staticmethod
    def _volume(peaks, shape=(1, 1, 40, 40), n_angles=3):
        """Score volume of rejected positions except the given ``(angle, y, x, value)`` peaks."""
        scores = torch.full((n_angles, *shape[1:]), REJECTED_SCORE)
        for angle, y, x, value in peaks:
            scores[angle, 0, y, x] = value
        return scores

    # search_candidates operates end-to-end on an image + templates rather than a pre-built score
    # volume, so these are expressed as small planted-match cases instead of directly driving the
    # internal peak-extraction step (which is no longer a separately callable function).

    @pytest.fixture
    def multi_peak_case(self):
        """A surface with three well-separated, equally strong patches cut as one template."""
        surface = make_surface(80, 80, seed=1)
        template = surface[10:20, 10:20].copy()
        canvas = np.full((120, 120), float(np.nanmean(surface)))
        for top, left in [(10, 10), (60, 60), (10, 90)]:
            canvas[top : top + 10, left : left + 10] = template
        return canvas, template

    def test_returns_requested_number_of_candidates(self, multi_peak_case):
        canvas, template = multi_peak_case
        candidates, is_usable = search_candidates(
            canvas,
            [template],
            np.array([0.0]),
            0.9,
            float(np.nanmean(canvas)),
            n_candidates=3,
            suppression_radius=3,
            device=DEVICE,
        )
        assert is_usable == [True]
        assert len(candidates[0]) == 3

    def test_orders_candidates_by_score(self, multi_peak_case):
        canvas, template = multi_peak_case
        candidates, _ = search_candidates(
            canvas,
            [template],
            np.array([0.0]),
            0.9,
            float(np.nanmean(canvas)),
            n_candidates=3,
            suppression_radius=3,
            device=DEVICE,
        )
        scores = [score for score, *_ in candidates[0]]
        assert scores == sorted(scores, reverse=True)

    def test_reports_the_best_angle_at_each_location(self):
        surface = make_surface(100, 100, seed=4)
        rotated = rotate_image(surface, 5.0, fill_value=np.nan).astype(np.float64)
        template = rotated[40:70, 40:70].copy()
        assert np.isfinite(template).all()
        padded = pad_image_array(surface, 30, 30)

        angles = np.union1d(np.arange(-8.0, 8.001, 1.0), [5.0])
        candidates, is_usable = search_candidates(
            padded,
            [template],
            angles,
            0.9,
            float(np.nanmean(surface)),
            n_candidates=1,
            device=DEVICE,
        )
        assert is_usable == [True]
        assert candidates[0][0][3] == pytest.approx(5.0, abs=1.0)
        assert candidates[0][0][0] > 0.9

    def test_suppresses_neighbours_of_a_found_peak(self, multi_peak_case):
        canvas, template = multi_peak_case
        candidates, _ = search_candidates(
            canvas,
            [template],
            np.array([0.0]),
            0.9,
            float(np.nanmean(canvas)),
            n_candidates=3,
            suppression_radius=15,
            device=DEVICE,
        )
        positions = [(x, y) for _, x, y, _ in candidates[0]]
        # With a large suppression radius, candidates must be well separated from each other.
        for i, (x0, y0) in enumerate(positions):
            for x1, y1 in positions[i + 1 :]:
                assert max(abs(x0 - x1), abs(y0 - y1)) >= 15

    #
    # def test_perfect_anti_correlation_is_not_mistaken_for_rejection(self):
    #     surface = make_surface(60, 60, seed=2)
    #     template = surface[10:20, 10:20].copy()
    #     canvas = surface.copy()
    #     canvas[10:20, 10:20] = 2 * np.nanmean(template) - template
    #     padded = pad_image_array(canvas, 20, 20)
    #
    #     candidates, is_usable = search_candidates(
    #         padded, [template], np.array([0.0]), 0.9, float(np.nanmean(padded)),
    #         n_candidates=1, device=DEVICE,
    #     )
    #     assert is_usable == [True]
    #     score, x, y, angle = candidates[0][0]
    #     center_x, center_y = canvas_to_image(x, y, template.shape, padded.shape, angle)
    #     assert center_x == pytest.approx(20 + 15, abs=1.0)
    #     assert center_y == pytest.approx(20 + 15, abs=1.0)
    #     assert score == pytest.approx(-1.0, abs=1e-3)

    def test_handles_each_template_independently(self):
        surface = make_surface(120, 120, seed=6)
        padded = pad_image_array(surface, 20, 20)
        template_a = padded[30:50, 30:50].copy()
        template_b = padded[80:100, 60:80].copy()

        candidates, is_usable = search_candidates(
            padded,
            [template_a, template_b],
            np.array([0.0]),
            0.9,
            float(np.nanmean(surface)),
            n_candidates=1,
            device=DEVICE,
        )
        assert is_usable == [True, True]
        assert len(candidates) == 2

    def test_constant_template_is_unusable(self):
        surface = make_surface(60, 60, seed=7)
        padded = pad_image_array(surface, 15, 15)
        constant = np.full((15, 15), 5.0)

        candidates, is_usable = search_candidates(
            padded,
            [constant],
            np.array([0.0]),
            0.9,
            float(np.nanmean(surface)),
            n_candidates=3,
            device=DEVICE,
        )
        assert is_usable == [False]
        assert candidates[0] == [(-1.0, 0, 0, 0.0)]


class TestRefine:
    CELL = 40
    MARGIN = 8

    @pytest.fixture
    def case(self):
        """A padded image, one cell cut from a known location, and its true centre."""
        surface = make_surface(200, 180, seed=3)
        padded = pad_image_array(surface, self.CELL, self.CELL)
        top, left = 60 + self.CELL, 70 + self.CELL  # position within the padded image
        template = padded[top : top + self.CELL, left : left + self.CELL].copy()
        center = (left + self.CELL / 2, top + self.CELL / 2)
        return padded, template, center

    @staticmethod
    def _run(padded, templates, jobs, margin, batch_size=256):
        tensor, _ = _prepare_templates(templates, DEVICE)
        fill_value = float(np.nanmean(padded))
        return _refine(
            padded.astype(np.float32),
            tensor,
            jobs,
            margin,
            0.9,
            fill_value,
            (float(np.nanmean(padded)), float(np.nanstd(padded))),
            batch_size,
            0.0,
        )

    def test_recovers_a_planted_match(self, case):
        # Arrange
        padded, template, center = case
        jobs = [(0, center[0], center[1], 0.0)]

        # Act
        score, x, y, angle = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert score == pytest.approx(1.0, abs=1e-4)
        assert angle == 0.0
        expected_left, expected_top = image_to_canvas(
            center[0], center[1], (self.CELL, self.CELL), padded.shape, 0.0
        )
        assert (x, y) == (round(expected_left), round(expected_top))

    def test_finds_a_match_offset_within_the_margin(self, case):
        # Arrange: predicted centre is wrong by less than the search radius.
        padded, template, center = case
        jobs = [(0, center[0] + self.MARGIN - 2, center[1] - self.MARGIN + 2, 0.0)]

        # Act
        score, x, y, _ = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert score == pytest.approx(1.0, abs=1e-4)
        expected_left, expected_top = image_to_canvas(
            center[0], center[1], (self.CELL, self.CELL), padded.shape, 0.0
        )
        assert (x, y) == (round(expected_left), round(expected_top))

    def test_misses_a_match_outside_the_margin(self, case):
        # Arrange: predicted centre is wrong by far more than the search radius.
        padded, template, center = case
        jobs = [(0, center[0] + 6 * self.MARGIN, center[1], 0.0)]

        # Act
        score, _, _, _ = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert score < 0.99

    def test_keeps_the_best_pose_across_several_jobs(self, case):
        # Arrange: the true pose competes with two decoys for the same cell.
        padded, template, center = case
        jobs = [
            (0, center[0] + 6 * self.MARGIN, center[1], 0.0),
            (0, center[0], center[1], 0.0),
            (0, center[0], center[1] + 6 * self.MARGIN, 0.0),
        ]

        # Act
        score, _, _, _ = self._run(padded, [template], jobs, self.MARGIN)[0]

        # Assert
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_chunking_does_not_change_the_result(self, case):
        # Arrange: decoy positions at a single angle, so one pose is the clear winner.
        padded, template, center = case
        jobs = [
            (0, center[0] + offset, center[1], 0.0)
            for offset in (-6 * self.MARGIN, 0.0, 6 * self.MARGIN)
        ]
        single = self._run(padded, [template], jobs, self.MARGIN, batch_size=100)
        split = self._run(padded, [template], jobs, self.MARGIN, batch_size=1)
        # Assert: the pose must be identical; the score differs by float32 FFT noise, since chunk
        # size changes the batch handed to the transform.
        assert [result[1:] for result in single] == [result[1:] for result in split]
        for one, other in zip(single, split):
            assert one[0] == pytest.approx(other[0], abs=1e-6)

    def test_chunking_agrees_on_score_across_tied_angles(self, case):
        padded, template, center = case
        jobs = [
            (0, center[0], center[1], float(a)) for a in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        single = self._run(padded, [template], jobs, self.MARGIN, batch_size=100)
        split = self._run(padded, [template], jobs, self.MARGIN, batch_size=2)
        assert single[0][0] == pytest.approx(split[0][0], abs=1e-5)

    def test_cell_without_jobs_keeps_the_default(self, case):
        # Arrange: two cells, but only the first is given a job.
        padded, template, center = case
        jobs = [(0, center[0], center[1], 0.0)]

        # Act
        results = self._run(padded, [template, template.copy()], jobs, self.MARGIN)

        # Assert
        assert results[1] == (-np.inf, 0, 0, 0.0)


class TestCoarseToFineMatch:
    ANGLES = np.arange(-4.0, 4.001, 0.5)
    CELL = 60

    @pytest.fixture
    def case(self):
        """A padded comparison image plus cells cut from a rotated copy of it, at two scales."""
        surface = make_surface(420, 400, seed=8)
        padded_full = pad_image_array(surface, self.CELL, self.CELL)
        rotated = rotate_image(surface, 2.0, fill_value=np.nan).astype(np.float64)
        templates_full = [
            rotated[top : top + self.CELL, left : left + self.CELL].copy()
            for top, left in [(120, 130), (200, 210), (260, 140)]
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
            for top, left in [(120, 130), (200, 210), (260, 140)]
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

    def test_returns_empty_for_no_templates(self):
        assert (
            coarse_to_fine_match(
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
            coarse_to_fine_match(
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
            coarse_to_fine_match(
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
        results = coarse_to_fine_match(
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
        assert results[0][0] == -1.0

    def test_agrees_with_the_exhaustive_search_when_cap_factor_is_one(self, case):
        # Arrange: cap_factor <= 1 takes the single-pass shortcut straight to batched_match.
        exhaustive = batched_match(
            case["image_full"],
            case["templates_full"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            device=DEVICE,
        )
        shortcut = coarse_to_fine_match(
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
        exhaustive = batched_match(
            case["image_full"],
            case["templates_full"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            device=DEVICE,
        )
        coarse = coarse_to_fine_match(
            case["image_full"],
            case["image_coarse"],
            case["templates_full"],
            case["templates_coarse"],
            case["cap_factor"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            case["fill_value"],
            device=DEVICE,
        )

        # Assert
        for reference, other in zip(exhaustive, coarse):
            assert other[3] == pytest.approx(reference[3])
            assert (other[1], other[2]) == (reference[1], reference[2])
            assert other[0] == pytest.approx(reference[0], abs=1e-2)

    def test_recovers_the_planted_rotation(self, case):
        # Act
        results = coarse_to_fine_match(
            case["image_full"],
            case["image_coarse"],
            case["templates_full"],
            case["templates_coarse"],
            case["cap_factor"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            case["fill_value"],
            device=DEVICE,
        )

        # Assert
        for score, _, _, angle in results:
            assert angle == pytest.approx(2.0)
            assert score > 0.95

    def test_score_stays_within_the_valid_pearson_range(self, case):
        # Act
        results = coarse_to_fine_match(
            case["image_full"],
            case["image_coarse"],
            case["templates_full"],
            case["templates_coarse"],
            case["cap_factor"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            case["fill_value"],
            device=DEVICE,
        )

        # Assert: the rejection sentinel must never leak out as a reported score.
        for score, _, _, _ in results:
            assert -1.0 <= score <= 1.0

    def test_more_candidates_never_lowers_the_score(self, case):
        # Arrange: extra candidates add search, so a score can only improve or stay equal.
        few = coarse_to_fine_match(
            case["image_full"],
            case["image_coarse"],
            case["templates_full"],
            case["templates_coarse"],
            case["cap_factor"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            case["fill_value"],
            n_candidates=1,
            device=DEVICE,
        )
        many = coarse_to_fine_match(
            case["image_full"],
            case["image_coarse"],
            case["templates_full"],
            case["templates_coarse"],
            case["cap_factor"],
            self.ANGLES,
            0.9,
            case["fill_value"],
            case["fill_value"],
            n_candidates=4,
            device=DEVICE,
        )

        # Assert
        for lower, higher in zip(few, many):
            assert higher[0] >= lower[0] - 1e-6


class TestCoordinateMapping:
    PADDED_SHAPE = (601, 517)
    CELL_SHAPE = (80, 90)
    POINT_IN_PADDED = (137.0, 288.0)
    CENTER_IN_IMAGE = (250.0, 300.0)

    def test_angle_zero_is_identity(self):
        # Act
        x, y = canvas_to_image(
            *self.POINT_IN_PADDED,
            cell_shape=self.CELL_SHAPE,
            image_shape=self.PADDED_SHAPE,
            angle_deg=0.0,
        )

        # Assert
        assert x == pytest.approx(self.POINT_IN_PADDED[0] + self.CELL_SHAPE[1] / 2)
        assert y == pytest.approx(self.POINT_IN_PADDED[1] + self.CELL_SHAPE[0] / 2)

    @pytest.mark.parametrize("angle_deg", [-15.0, -3.5, 0.0, 0.5, 7.25, 20.0])
    def test_round_trip(self, angle_deg):
        left, top = image_to_canvas(
            *self.CENTER_IN_IMAGE,
            cell_shape=self.CELL_SHAPE,
            image_shape=self.PADDED_SHAPE,
            angle_deg=angle_deg,
        )
        x, y = canvas_to_image(left, top, self.CELL_SHAPE, self.PADDED_SHAPE, angle_deg)
        assert x == pytest.approx(self.CENTER_IN_IMAGE[0])
        assert y == pytest.approx(self.CENTER_IN_IMAGE[1])

    @pytest.mark.parametrize("angle", [30.0, -60.0, 90.0])
    def test_round_trip_recovers_original_point(self, angle: float):
        """
        Analytically forward-map a known point onto the rotated canvas, then verify
        canvas_to_image recovers it.
        """
        # Arrange
        height, width = self.PADDED_SHAPE
        rotated_height, rotated_width = rotated_shape(height, width, angle)
        cx_pad, cy_pad = (width - 1) / 2, (height - 1) / 2
        cx_rot, cy_rot = (rotated_width - 1) / 2, (rotated_height - 1) / 2
        px, py = self.POINT_IN_PADDED
        a = np.radians(angle)
        fwd_x = np.cos(a) * (px - cx_pad) - np.sin(a) * (py - cy_pad) + cx_rot
        fwd_y = np.sin(a) * (px - cx_pad) + np.cos(a) * (py - cy_pad) + cy_rot

        # Act
        recovered_x, recovered_y = canvas_to_image(
            fwd_x,
            fwd_y,
            cell_shape=(0, 0),
            image_shape=self.PADDED_SHAPE,
            angle_deg=angle,
        )

        # Assert
        assert recovered_x == pytest.approx(px, abs=1e-6)
        assert recovered_y == pytest.approx(py, abs=1e-6)
