import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import pytest
from conversion.plots.plot_score_llr_transformation import plot_loglr_with_confidence

from ..helper_functions import assert_plot_is_valid_image


def verify_plot_properties(
    ax: Axes, expected_num_lines: int, should_have_llr_label: bool
):
    assert len(ax.lines) == expected_num_lines
    assert ax.get_xlabel() == "Score"
    assert ax.get_ylabel() == "LogLR"
    assert ax.get_title() == "LogLR plot (with confidence intervals)"

    # Verify legend entries
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
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
        score = np.linspace(0, 1, n_points)
        llr = 5 * (score - 0.5) ** 2 - 2
        llr_5 = llr - 0.5
        llr_95 = llr + 0.5

        return {"score": score, "llr": llr, "5% llr": llr_5, "95% llr": llr_95}

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
        sample_data: dict[str, np.ndarray],
        score_llr_point: tuple[float, float],
        expected_num_lines: int,
        should_have_llr_label: bool,
    ):
        """Test plotting with and without score_llr_point."""
        fig, ax = plt.subplots()

        # Should not raise any exceptions
        plot_loglr_with_confidence(ax, sample_data, score_llr_point)

        verify_plot_properties(
            ax, expected_num_lines, should_have_llr_label
        )  # Verify that plot was created
        assert_plot_is_valid_image(fig, tmp_path)

        plt.close(fig)

    def test_missing_keys_raises_error(self):
        """Test that missing required keys raises ValueError."""
        fig, ax = plt.subplots()

        incomplete_data = {
            "score": np.linspace(0, 1, 100),
            "llr": np.random.randn(100),
            # Missing '5% llr' and '95% llr'
        }

        with pytest.raises(ValueError) as exc_info:
            plot_loglr_with_confidence(ax, incomplete_data, None)

        assert "Missing required keys" in str(exc_info.value)
        assert "'5% llr'" in str(exc_info.value) or "5% llr" in str(exc_info.value)
        assert "'95% llr'" in str(exc_info.value) or "95% llr" in str(exc_info.value)

        plt.close(fig)

    def test_all_keys_missing_raises_error(self):
        """Test that completely empty dict raises ValueError."""
        fig, ax = plt.subplots()

        with pytest.raises(ValueError):
            plot_loglr_with_confidence(ax, {}, None)

        plt.close()
