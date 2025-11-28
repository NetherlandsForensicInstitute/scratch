from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_test_data(data) -> Figure:
    """Plot test data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")
    return fig
