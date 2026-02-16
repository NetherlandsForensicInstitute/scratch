from matplotlib.axes import Axes

from conversion.plots.data_formats import LlrTransformationData


def plot_score_llr_transformation(ax: Axes, data: LlrTransformationData) -> None:
    """
    Plot LogLR curve with confidence intervals.

    :param ax: Matplotlib axes object to plot on.
    :param data: LLR transformation data containing scores, llrs, confidence intervals,
        and optional score_llr_point.
    """

    # Plot main LogLR curve
    ax.plot(data.scores, data.llrs, "b-", linewidth=2, label="LogLR all")

    # Plot confidence intervals with different dash styles
    ax.plot(
        data.scores,
        data.llrs_at5,
        "b--",
        linewidth=1,
        dashes=(5, 3),
        label="LogLR all 5%",
    )
    ax.plot(
        data.scores,
        data.llrs_at95,
        "b--",
        linewidth=1,
        dashes=(2, 2),
        label="LogLR all 95%",
    )

    # Set labels and title
    ax.set_xlabel("Score")
    ax.set_ylabel("LogLR")
    ax.set_title("LogLR plot (with confidence intervals)")

    # Set grid
    ax.grid(True, alpha=0.3)

    # Adjust y-axis to show the full range (do this before drawing the coordinate lines)
    y_min = min(data.llrs.min(), data.llrs_at5.min())
    y_max = max(data.llrs.max(), data.llrs_at95.max())
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlim(data.scores.min(), data.scores.max())

    # Add horizontal and vertical lines at the score_llr_point
    if data.score_llr_point:
        score_point, llr_point = data.score_llr_point
        # Horizontal line from y-axis to the score point
        ax.plot(
            [data.scores.min(), score_point],
            [llr_point, llr_point],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="LogLR",
        )
        # Vertical line from x-axis to the llr point
        ax.plot(
            [score_point, score_point],
            [y_min - y_margin, llr_point],
            color="green",
            linestyle="--",
            linewidth=1.5,
        )

    # Add legend
    ax.legend(loc="upper left", frameon=True)
