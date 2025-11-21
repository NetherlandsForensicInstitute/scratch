import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison
from numpy._typing import NDArray

from image_generation import get_surface_map
from image_generation.data_formats import Image2DArray
from parsers.data_types import ScanImage
from utils.paths import ROOT_DIR


@pytest.fixture
def mask(scan_image: ScanImage) -> NDArray[tuple[int, int]]:
    """Makes a mask used for masking that area (all is 1 except the drawn square)."""
    mask = np.ones((scan_image.width, scan_image.height), dtype=np.uint8)
    square_size = 85
    x_start_cor = 100
    y_start_cor = 150
    mask[
        x_start_cor : x_start_cor + square_size, y_start_cor : y_start_cor + square_size
    ] = 0
    return mask


def plot_test_data(data, show_plot=True) -> Figure:
    """Plot test data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")
    if show_plot:
        fig.show()
    return fig


@pytest.fixture
def mask_crop(scan_image: ScanImage) -> NDArray[tuple[int, int]]:
    """Makes a mask used for cropping (all is 0 except the drawn square)."""
    mask = np.zeros((scan_image.width, scan_image.height), dtype=np.uint8)
    square_size = 85
    x_start_cor = 100
    y_start_cor = 150
    mask[
        x_start_cor : x_start_cor + square_size, y_start_cor : y_start_cor + square_size
    ] = 1
    return mask


@pytest.fixture
def data_in(scan_image: ScanImage) -> ScanImage:
    return ScanImage.from_file(
        ROOT_DIR / "tests/resources/scans/Klein_non_replica_mode.al3d"
    )


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot_default"], extensions=["png"])
def test_get_surface_plot(data_in: ScanImage) -> None:
    data = get_surface_map(
        depth_data=Image2DArray(data=data_in.data),
        x_dimension=data_in.scale_x,
        y_dimension=data_in.scale_y,
    )
    plot_test_data(data.data, show_plot=False)
