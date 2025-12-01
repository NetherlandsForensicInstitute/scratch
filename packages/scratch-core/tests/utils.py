from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_test_data(data: NDArray) -> Figure:
    """Plot 2D image data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")
    return fig
