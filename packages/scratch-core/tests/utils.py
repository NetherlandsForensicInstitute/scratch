from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from utils.array_definitions import ScanMap2DArray


def plot_test_data(data: ScanMap2DArray) -> Figure:
    """Plot 2D image data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")
    return fig
