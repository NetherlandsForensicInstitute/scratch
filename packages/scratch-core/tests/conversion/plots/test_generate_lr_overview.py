"""
Generate the LR overview image with synthetic data. Will be removed before merging.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from conversion.data_formats import Mark
from conversion.plots.data_formats import ImpressionComparisonMetrics
from conversion.plots.plot_impression import _plot_cell_overlay_on_axes
from conversion.plots.plot_score_histogram_kde import plot_score_histograms_kde
from conversion.plots.score_llr_transformation_plot import plot_loglr_with_confidence
from conversion.plots.utils import (
    draw_metadata_box,
    figure_to_array,
    get_height_ratios,
    get_metadata_dimensions,
)

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
        """Produce lr_overview.png and verify it is a valid RGB image."""
        wrap_width = 25
        metrics = impression_overview_metrics
        mark_reference_filtered = impression_overview_marks["reference_filtered"]
        mark_compared_filtered = impression_overview_marks["compared_filtered"]

        metadata_reference = impression_overview_metadata_reference
        metadata_compared = impression_overview_metadata_compared

        n_cell_rows, n_cell_cols = metrics.cell_correlations.shape
        n_cells = n_cell_rows * n_cell_cols
        n_cmc = int(
            np.sum(metrics.cell_correlations >= metrics.cell_similarity_threshold)
        )
        cmc_score = n_cmc / n_cells * 100 if n_cells > 0 else 0.0

        results_items: dict[str, str] = {
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
        km_scores = rng.normal(loc=28, scale=5, size=n_km)
        km_scores = np.clip(km_scores, 0, None)
        scores = np.concatenate([knm_scores, km_scores])
        labels = np.concatenate([np.zeros(n_knm), np.ones(n_km)])

        n_points = 200
        llr_score = np.linspace(0, 55, n_points)
        llr = np.piecewise(
            llr_score,
            [llr_score < 20, llr_score >= 20],
            [lambda s: -2 + 0.1 * s, lambda s: -2 + 0.35 * (s - 10)],
        )
        llr_data = {
            "score": llr_score,
            "llr": llr,
            "5% llr": llr - 0.3,
            "95% llr": llr + 0.3,
        }
        score_llr_point = (cmc_score, float(np.interp(cmc_score, llr_score, llr)))

        max_metadata_rows, metadata_height_ratio = get_metadata_dimensions(
            metadata_compared, metadata_reference, wrap_width
        )
        height_ratios = get_height_ratios(metadata_height_ratio, 0.40, 0.40)

        fig_height = 12 + (max_metadata_rows * 0.12)
        fig_height = max(10.0, min(15.0, fig_height))

        fig = plt.figure(figsize=(16, fig_height))

        gs = fig.add_gridspec(
            3,
            3,
            height_ratios=height_ratios,
            width_ratios=[0.35, 0.35, 0.30],
            hspace=0.35,
            wspace=0.45,
        )

        gs_meta = gs[0, :].subgridspec(1, 2, wspace=0.15)
        ax_meta_ref = fig.add_subplot(gs_meta[0, 0])
        draw_metadata_box(
            ax_meta_ref,
            metadata_reference,
            "Reference Surface (A)",
            wrap_width=wrap_width,
        )
        ax_meta_comp = fig.add_subplot(gs_meta[0, 1])
        draw_metadata_box(
            ax_meta_comp,
            metadata_compared,
            "Compared Surface (B)",
            wrap_width=wrap_width,
        )

        ax_filtered_ref = fig.add_subplot(gs[1, 0])
        im_ref = _plot_cell_overlay_on_axes(
            ax_filtered_ref,
            mark_reference_filtered.scan_image.data,
            mark_reference_filtered.scan_image.scale_x,
            metrics.cell_correlations,
            cell_label_prefix="A",
            cell_similarity_threshold=metrics.cell_similarity_threshold,
            show_all_cells=True,
        )
        ax_filtered_ref.set_title(
            "Filtered Reference Surface A", fontsize=12, fontweight="bold"
        )
        divider_ref = make_axes_locatable(ax_filtered_ref)
        cax_ref = divider_ref.append_axes("right", size="5%", pad=0.05)
        cbar_ref = fig.colorbar(im_ref, cax=cax_ref, label="Scan Depth [µm]")
        cbar_ref.ax.tick_params(labelsize=9)

        ax_filtered_comp = fig.add_subplot(gs[1, 1])
        ref_h, ref_w = mark_reference_filtered.scan_image.data.shape
        ref_scale = mark_reference_filtered.scan_image.scale_x
        n_rows, n_cols = metrics.cell_correlations.shape
        cell_size_um = (
            ref_w * ref_scale * 1e6 / n_cols,
            ref_h * ref_scale * 1e6 / n_rows,
        )
        im_comp = _plot_cell_overlay_on_axes(
            ax_filtered_comp,
            mark_compared_filtered.scan_image.data,
            mark_compared_filtered.scan_image.scale_x,
            metrics.cell_correlations,
            cell_label_prefix="B",
            cell_similarity_threshold=metrics.cell_similarity_threshold,
            show_all_cells=False,
            cell_positions=metrics.cell_positions_compared,
            cell_rotations=metrics.cell_rotations_compared,
            cell_size_um=cell_size_um
            if metrics.cell_positions_compared is not None
            else None,
        )
        ax_filtered_comp.set_title(
            "Filtered, Moved Compared Surface B", fontsize=12, fontweight="bold"
        )
        divider_comp = make_axes_locatable(ax_filtered_comp)
        cax_comp = divider_comp.append_axes("right", size="5%", pad=0.05)
        cbar_comp = fig.colorbar(im_comp, cax=cax_comp, label="Scan Depth [µm]")
        cbar_comp.ax.tick_params(labelsize=9)

        ax_results = fig.add_subplot(gs[1, 2])
        draw_metadata_box(
            ax_results,
            results_items,
            draw_border=False,
            wrap_width=wrap_width,
        )

        gs_bottom = gs[2, :].subgridspec(1, 2, wspace=0.30)
        ax_hist = fig.add_subplot(gs_bottom[0, 0])
        plot_score_histograms_kde(scores, labels, ax_hist, new_score=cmc_score)
        ax_hist.set_title("Score histograms", fontsize=12, fontweight="bold")

        ax_llr = fig.add_subplot(gs_bottom[0, 1])
        plot_loglr_with_confidence(ax_llr, llr_data, score_llr_point)

        fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
        fig.subplots_adjust(left=0.06, right=0.93, top=0.96, bottom=0.06)

        overview = figure_to_array(fig)
        plt.close(fig)

        assert_valid_rgb_image(overview)

        out = Path(__file__).resolve().parents[5] / "lr_overview.png"
        Image.fromarray(overview).save(out)
