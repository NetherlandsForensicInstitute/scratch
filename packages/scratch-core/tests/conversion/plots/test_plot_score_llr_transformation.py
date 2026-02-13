import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import pytest

from container_models.base import FloatArray1D
from conversion.plots.plot_score_llr_transformation import plot_score_llr_transformation

from ..helper_functions import assert_plot_is_valid_image


def verify_plot_properties(
    ax: Axes, expected_num_lines: int, should_have_llr_label: bool
):
    assert len(ax.lines) == expected_num_lines
    assert ax.get_xlabel() == "Score"
    assert ax.get_ylabel() == "LogLR"
    assert ax.get_title() == "LogLR plot (with confidence intervals)"

    # Verify legend entries
    legend = ax.get_legend()
    if legend:
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert "LogLR all" in legend_labels
        assert "LogLR all 5%" in legend_labels
        assert "LogLR all 95%" in legend_labels

        if should_have_llr_label:
            assert "LogLR" in legend_labels
        else:
            assert "LogLR" not in legend_labels


class TestPlotLoglrWithConfidence:
    """Test suite for plot_loglr_with_confidence function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_points = 100
        scores = np.linspace(0, 1, n_points)
        llrs = 5 * (scores - 0.5) ** 2 - 2
        llrs_at5 = llrs - 0.5
        llrs_at95 = llrs + 0.5

        return scores, llrs, llrs_at5, llrs_at95

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "score_llr_point, expected_num_lines, should_have_llr_label",
        [
            ((0.9, -1.2), 5, True),  # With score_llr_point
            (None, 3, False),  # Without score_llr_point
        ],
        ids=["with_score_llr_point", "without_score_llr_point"],
    )
    def test_plot_with_and_without_score_llr_point(
        self,
        tmp_path: Path,
        sample_data: tuple[FloatArray1D, FloatArray1D, FloatArray1D, FloatArray1D],
        score_llr_point: tuple[float, float],
        expected_num_lines: int,
        should_have_llr_label: bool,
    ):
        (scores, llrs, llrs_at5, llrs_at95) = sample_data
        """Test plotting with and without score_llr_point."""
        fig, ax = plt.subplots()

        # Should not raise any exceptions
        plot_score_llr_transformation(
            ax, scores, llrs, llrs_at5, llrs_at95, score_llr_point
        )

        verify_plot_properties(
            ax, expected_num_lines, should_have_llr_label
        )  # Verify that plot was created
        assert_plot_is_valid_image(fig, tmp_path)

        plt.close(fig)
