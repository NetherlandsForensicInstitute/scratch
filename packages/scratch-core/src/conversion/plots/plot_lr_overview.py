import matplotlib.pyplot as plt

from container_models.base import ImageRGB
from conversion.plots.data_formats import LlrTransformationData
from conversion.plots.plot_score_llr_transformation import plot_score_llr_transformation
from conversion.plots.utils import figure_to_array


def plot_lr_overview(data: LlrTransformationData) -> ImageRGB:
    """Generate the LR overview plot as an RGB image.

    :param data: LLR transformation data.
    :returns: RGB image as uint8 array.
    """
    fig, ax = plt.subplots()
    plot_score_llr_transformation(ax, data)
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr
