"""Tests for conversion.surface_comparison.cell_registration.stage_builders."""

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.stage_builders import (
    build_coarse_stage,
    compute_cap_factor,
)
from conversion.surface_comparison.pipeline import resample_scan_image_nan_aware

from .helpers import make_grid_cell

PIXEL_SIZE = 1e-6


class TestComputeCapFactor:
    def test_rejects_a_pixel_scale_mismatch(self):
        # Arrange: the pipeline is responsible for bringing both images onto one grid before
        # their pixel counts get compared against coarse_target_size.
        reference_image = ScanImage(
            data=np.zeros((80, 80)), scale_x=PIXEL_SIZE, scale_y=PIXEL_SIZE
        )
        comparison_image = reference_image.model_copy(
            update={"scale_x": PIXEL_SIZE * 2}
        )

        # Act / Assert
        with pytest.raises(ValueError, match="same pixel scale"):
            compute_cap_factor(
                reference_image, comparison_image, 20, 20, coarse_target_size=256
            )


class TestBuildCoarseStage:
    def test_rejects_grid_cells_with_mixed_nan_fill_values(self):
        # Arrange: the coarse stage picks each cell's candidate locations, so templates filled
        # inconsistently would change where cells match, not just how the match is polished.
        rng = np.random.default_rng(2)
        image = ScanImage(
            data=rng.random((120, 120)), scale_x=PIXEL_SIZE, scale_y=PIXEL_SIZE
        )
        grid_cells = [
            make_grid_cell(data=image.data[10:30, 10:30], nan_fill_value=0.0),
            make_grid_cell(data=image.data[40:60, 40:60], nan_fill_value=1.0),
        ]

        # Act / Assert
        with pytest.raises(ValueError, match="nan_fill_value"):
            build_coarse_stage(image, image, grid_cells, cap_factor=3.0)

    def test_coarse_canvas_lands_on_the_reference_scale_regardless_of_comparison_source(
        self,
    ):
        """
        Regression test: the comparison canvas must land on the same physical coarse grid as
        the reference, whether the comparison image passed in is the original scan (its own
        native pixel scale) or one already resampled onto the reference's scale.

        Previously, downsampling the *original* comparison image by ``cap_factor`` alone (no
        adjustment for its native scale relative to the reference) produced a coarse canvas at
        the wrong physical pixel pitch whenever the two scans had different native resolutions.
        """
        # Arrange: reference and comparison cover the same physical scene, but the comparison
        # scan's native pixel pitch is half the reference's (twice the pixel density).
        rng = np.random.default_rng(0)
        reference_image = ScanImage(
            data=rng.random((200, 200)).astype(np.float64), scale_x=1.0, scale_y=1.0
        )
        comparison_original = ScanImage(
            data=rng.random((400, 400)).astype(np.float64), scale_x=0.5, scale_y=0.5
        )
        comparison_aligned = resample_scan_image_nan_aware(
            comparison_original, reference_image.scale_x
        )
        grid_cell = make_grid_cell(
            data=reference_image.data[50:70, 50:70], top_left=(50, 50)
        )
        cap_factor = 5 / 3

        # Act
        from_original = build_coarse_stage(
            comparison_original, reference_image, [grid_cell], cap_factor
        )
        from_aligned = build_coarse_stage(
            comparison_aligned, reference_image, [grid_cell], cap_factor
        )

        # Assert
        assert from_original.image.shape == from_aligned.image.shape

    def test_coarse_canvas_matches_cap_factor_when_already_aligned(self):
        # Arrange: comparison already on the reference's scale, so scale_match_factor == 1 and
        # this reduces to a plain downsample by cap_factor, as before.
        rng = np.random.default_rng(1)
        reference_image = ScanImage(
            data=rng.random((120, 120)).astype(np.float64), scale_x=1.0, scale_y=1.0
        )
        comparison_image = ScanImage(
            data=rng.random((120, 120)).astype(np.float64), scale_x=1.0, scale_y=1.0
        )
        grid_cell = make_grid_cell(
            data=reference_image.data[10:30, 10:30], top_left=(10, 10)
        )
        cap_factor = 3.0

        # Act
        coarse = build_coarse_stage(
            comparison_image, reference_image, [grid_cell], cap_factor
        )

        # Assert
        expected_side = max(1, int(np.ceil(120 / cap_factor)))
        coarse_cell = max(1, int(np.ceil(grid_cell.width / cap_factor)))
        assert coarse.image.shape == (
            expected_side + 2 * coarse_cell,
            expected_side + 2 * coarse_cell,
        )
