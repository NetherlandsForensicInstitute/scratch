from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.plots.data_formats import (
    StriationComparisonMetrics,
    ImpressionComparisonMetrics,
)
from conversion.plots.plot_impression import plot_impression_comparison_results
from conversion.plots.plot_striation import plot_striation_comparison_results

from .helper_functions import (
    assert_valid_rgb_image,
    create_synthetic_profile_mark,
    create_synthetic_striation_mark,
)


def _make_base_pattern(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / cols
    yn = y / rows

    surface = (
        2.0 * np.sin(2 * np.pi * yn * 8)
        + 1.2 * np.sin(2 * np.pi * yn * 14 + 1.0)
        + 0.7 * np.cos(2 * np.pi * yn * 22)
        + 0.4 * np.sin(2 * np.pi * yn * 35 + 0.3)
    )
    surface *= 1.0 + 0.15 * np.sin(2 * np.pi * xn * 2 + 0.7)
    surface += 1.0 * (yn - 0.5) + 0.5 * ((xn - 0.5) ** 2)

    for _ in range(6):
        cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        amp = rng.uniform(-1.5, 1.5)
        sigma = rng.uniform(0.03, 0.08)
        surface += amp * np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / (2 * sigma**2))

    return surface


def _make_surface_pair(
    rows: int, cols: int, base_seed: int, noise_seed_a: int, noise_seed_b: int
) -> tuple[np.ndarray, np.ndarray]:
    base = _make_base_pattern(rows, cols, seed=base_seed)
    noise_a = np.random.default_rng(noise_seed_a).normal(0, 0.25, (rows, cols))
    noise_b = np.random.default_rng(noise_seed_b).normal(0, 0.25, (rows, cols))
    return (
        ((base + noise_a) * 1e-6).astype(np.float64),
        ((base + noise_b) * 1e-6).astype(np.float64),
    )


@pytest.mark.integration
class TestGenerateOverview:
    """Generate the full comparison overview with realistic synthetic data."""

    def test_generates_overview_png(self) -> None:
        """Produce plot_results_overview.png and verify it is a valid RGB image."""
        rows, cols = 300, 200
        scale_x = 1.5626e-6
        scale_y = 1.5675e-6

        data_ref_lev, data_comp_lev = _make_surface_pair(rows, cols, 0, 10, 11)
        data_ref_flt, data_comp_flt = _make_surface_pair(rows, cols, 1, 12, 13)

        def _mark(data: np.ndarray) -> Mark:
            return Mark(
                scan_image=ScanImage(data=data, scale_x=scale_x, scale_y=scale_y),
                mark_type=MarkType.EJECTOR_IMPRESSION,
            )

        mark_ref_lev = _mark(data_ref_lev)
        mark_comp_lev = _mark(data_comp_lev)
        mark_ref_flt = _mark(data_ref_flt)
        mark_comp_flt = _mark(data_comp_flt)

        # Cell correlations — 3x4 grid, 2 CMC cells
        cell_similarity_threshold = 0.25
        cell_correlations = np.array(
            [
                [0.18, 0.09, 0.52, 0.14],
                [0.07, 0.11, 0.06, 0.41],
                [0.13, 0.22, 0.10, 0.05],
            ],
            dtype=np.float64,
        )

        n_cell_rows, n_cell_cols = cell_correlations.shape
        n_cells = n_cell_rows * n_cell_cols
        n_cmc = int(np.sum(cell_correlations >= cell_similarity_threshold))
        cmc_score = n_cmc / n_cells * 100

        surface_w_um = cols * scale_x * 1e6
        surface_h_um = rows * scale_y * 1e6
        cell_w_um = surface_w_um / n_cell_cols
        cell_h_um = surface_h_um / n_cell_rows

        rng = np.random.default_rng(42)
        positions = np.full((n_cells, 2), np.nan, dtype=np.float64)
        rotations = np.full(n_cells, np.nan, dtype=np.float64)
        global_dx, global_dy = 8.0, -6.0
        global_angle = np.deg2rad(2.5)

        for r in range(n_cell_rows):
            for c in range(n_cell_cols):
                if cell_correlations[r, c] < cell_similarity_threshold:
                    continue
                flat = r * n_cell_cols + c
                grid_cx = (c + 0.5) * cell_w_um
                grid_cy = (n_cell_rows - 1 - r + 0.5) * cell_h_um
                positions[flat, 0] = grid_cx + global_dx + rng.normal(0, 3)
                positions[flat, 1] = grid_cy + global_dy + rng.normal(0, 3)
                angle_noise = rng.normal(0, np.deg2rad(1.5)) * (
                    1 - cell_correlations[r, c]
                )
                rotations[flat] = global_angle + angle_noise

        metrics = ImpressionComparisonMetrics(
            area_correlation=0.4123,
            cell_correlations=cell_correlations,
            cmc_score=cmc_score,
            sq_ref=0.1234,
            sq_comp=0.1456,
            sq_diff=0.0567,
            has_area_results=True,
            has_cell_results=True,
            cell_positions_compared=positions,
            cell_rotations_compared=rotations,
            cell_similarity_threshold=cell_similarity_threshold,
            cmc_area_fraction=16.04,
            cutoff_low_pass=5.0,
            cutoff_high_pass=250.0,
            cell_size_um=125.0,
            max_error_cell_position=75.0,
            max_error_cell_angle=6.0,
        )

        metadata_ref = {
            "Collection/Case": "firearms",
            "Firearm ID": "firearm_1_known_match",
            "Specimen ID": "kras_1",
            "Measurement ID": "afsloeting_1",
        }
        metadata_comp = {
            "Collection/Case": "firearms",
            "Firearm ID": "firearm_1_known_match",
            "Specimen ID": "kras_2_r01",
            "Measurement ID": "afsloeting_2",
        }

        results = plot_impression_comparison_results(
            mark_reference_leveled=mark_ref_lev,
            mark_compared_leveled=mark_comp_lev,
            mark_reference_filtered=mark_ref_flt,
            mark_compared_filtered=mark_comp_flt,
            metrics=metrics,
            metadata_reference=metadata_ref,
            metadata_compared=metadata_comp,
        )

        overview = results.comparison_overview
        assert_valid_rgb_image(overview)

        # Save to project root so the image can be inspected after the test run
        out = Path(__file__).resolve().parents[5] / "plot_results_overview.png"
        Image.fromarray(overview).save(out)

    def test_generates_striation_overview_png(
        self,
        sample_metadata_reference: dict[str, str],
        sample_metadata_compared: dict[str, str],
    ) -> None:
        """Produce plot_striation_overview.png and verify it is a valid RGB image."""
        scale = 1.5625e-6

        mark_reference = create_synthetic_striation_mark(height=256, width=200, seed=42)
        mark_compared = create_synthetic_striation_mark(height=256, width=220, seed=43)
        mark_reference_aligned = create_synthetic_striation_mark(
            height=200, width=200, seed=44
        )
        mark_compared_aligned = create_synthetic_striation_mark(
            height=200, width=200, seed=45
        )
        mark_profile_reference = create_synthetic_profile_mark(length=200, seed=46)
        mark_profile_compared = create_synthetic_profile_mark(length=200, seed=47)

        quality_passbands: dict[tuple[float, float], float] = {
            (5, 250): 0.85,
            (100, 250): 0.78,
            (50, 100): 0.65,
            (25, 50): 0.45,
            (10, 25): 0.30,
            (5, 10): 0.15,
        }

        metrics = StriationComparisonMetrics(
            score=0.85,
            shift=12.5,
            overlap=80.4,
            sq_ref=0.2395,
            sq_comp=0.7121,
            sq_diff=0.6138,
            sq_ratio=297.3765,
            sign_diff_dsab=220.94,
            data_spacing=scale * 1e6,
            quality_passbands=quality_passbands,
        )

        results = plot_striation_comparison_results(
            mark_reference=mark_reference,
            mark_compared=mark_compared,
            mark_reference_aligned=mark_reference_aligned,
            mark_compared_aligned=mark_compared_aligned,
            mark_profile_reference_aligned=mark_profile_reference,
            mark_profile_compared_aligned=mark_profile_compared,
            metrics=metrics,
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
        )

        overview = results.comparison_overview
        assert_valid_rgb_image(overview)
