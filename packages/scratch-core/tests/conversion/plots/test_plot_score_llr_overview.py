"""
Generate the LR overview image with synthetic data. Will be removed before merging.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conversion.data_formats import Mark
from conversion.plots.data_formats import ImpressionComparisonMetrics
from conversion.plots.plot_cmc_comparison_overview import plot_cmc_comparison_overview

from .helper_functions import assert_valid_rgb_image


@pytest.mark.integration
class TestGenerateLROverview:
    """Generate the full LR overview image with realistic synthetic data."""

    def test_generates_lr_overview_png(
        self,
        impression_overview_marks: dict[str, Mark],
        impression_overview_metrics: ImpressionComparisonMetrics,
        impression_overview_metadata_reference: dict[str, str],
        impression_overview_metadata_compared: dict[str, str],
    ) -> None:
        """Produce lr_ACCF_overview.png and verify it is a valid RGB image."""
        metrics = impression_overview_metrics

        n_cell_rows, n_cell_cols = metrics.cell_correlations.shape
        n_cells = n_cell_rows * n_cell_cols
        n_cmc = int(
            np.sum(metrics.cell_correlations >= metrics.cell_similarity_threshold)
        )
        cmc_score = n_cmc / n_cells * 100 if n_cells > 0 else 0.0

        results_metadata: dict[str, str] = {
            "Date report": "2023-02-16",
            "User ID": "RUHES (apc_abai)",
            "Mark type": "Breech face impression",
            "Collection name": "glock_comparison_study_scrat...",
            "KM model": "Beta-binomial",
            "KNM model": "Binomial",
            "Score type": "CMC",
            "Score (transform)": f"{n_cmc} of {n_cells}",
            "LogLR (5%, 95%)": "4.87 (4.87, 4.87)",
            "LR (5%, 95%)": "7.41e+04 (7.41e+04, 7.41e+04)",
            "# of KM scores": "1125",
            "# of KNM scores": "171991",
        }

        rng = np.random.default_rng(42)
        n_knm, n_km = 171991, 1125
        knm_scores = rng.exponential(scale=2.0, size=n_knm)
        km_scores = np.clip(rng.normal(loc=28, scale=5, size=n_km), 0, None)
        scores = np.concatenate([knm_scores, km_scores])
        labels = np.concatenate([np.zeros(n_knm), np.ones(n_km)])

        llr_scores = np.linspace(0, 55, 200)
        llrs = np.piecewise(
            llr_scores,
            [llr_scores < 20, llr_scores >= 20],
            [lambda s: -2 + 0.1 * s, lambda s: -2 + 0.35 * (s - 10)],
        )
        score_llr_point = (cmc_score, float(np.interp(cmc_score, llr_scores, llrs)))

        overview = plot_cmc_comparison_overview(
            mark_reference_filtered=impression_overview_marks["reference_filtered"],
            mark_compared_filtered=impression_overview_marks["compared_filtered"],
            metrics=metrics,
            metadata_reference=impression_overview_metadata_reference,
            metadata_compared=impression_overview_metadata_compared,
            results_metadata=results_metadata,
            scores=scores,
            labels=labels,
            bins=20,
            densities=None,
            new_score=cmc_score,
            llr_scores=llr_scores,
            llrs=llrs,
            llrs_at5=llrs - 0.3,
            llrs_at95=llrs + 0.3,
            score_llr_point=score_llr_point,
        )

        assert_valid_rgb_image(overview)

        out = Path(__file__).resolve().parents[5] / "lr_CMC_overview.png"
        Image.fromarray(overview).save(out)
