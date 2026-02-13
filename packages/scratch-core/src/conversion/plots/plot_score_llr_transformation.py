from matplotlib.axes import Axes

from container_models.base import FloatArray1D


def plot_score_llr_transformation(
    ax: Axes,
    scores: FloatArray1D,
    llrs: FloatArray1D,
    llrs_at5: FloatArray1D,
    llrs_at95: FloatArray1D,
    score_llr_point: tuple[float, float] | None,
) -> None:
    """
    Plot LogLR curve with confidence intervals.

    :param ax: Matplotlib axes object to plot on.
    :param data: Dictionary containing score and LogLR arrays. Must have keys:
        'score', 'llr', '5% llr', and '95% llr'. All arrays must be 1D
        and of the same length.
    :param score_llr_point: Optional (score, llr) coordinate to mark on the plot
        with a horizontal green line. If None, no line is drawn.
    :raises ValueError: If data dict does not contain all required keys.

    """

    # Plot main LogLR curve
    ax.plot(scores, llrs, "b-", linewidth=2, label="LogLR all")

    # Plot confidence intervals with different dash styles
    ax.plot(scores, llrs_at5, "b--", linewidth=1, dashes=(5, 3), label="LogLR all 5%")
    ax.plot(scores, llrs_at95, "b--", linewidth=1, dashes=(2, 2), label="LogLR all 95%")

    # Set labels and title
    ax.set_xlabel("Score")
    ax.set_ylabel("LogLR")
    ax.set_title("LogLR plot (with confidence intervals)")

    # Set grid
    ax.grid(True, alpha=0.3)

    # Adjust y-axis to show the full range (do this before drawing the coordinate lines)
    y_min = min(llrs.min(), llrs_at5.min())
    y_max = max(llrs.max(), llrs_at95.max())
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlim(scores.min(), scores.max())

    # Add horizontal and vertical lines at the score_llr_point
    if score_llr_point:
        score_point, llr_point = score_llr_point
        # Horizontal line from y-axis to the score point
        ax.plot(
            [scores.min(), score_point],
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
