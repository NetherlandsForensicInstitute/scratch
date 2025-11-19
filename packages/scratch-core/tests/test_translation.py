import numpy as np
import pytest

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison
from numpy._typing import NDArray
from parsers.data_types import ScanImage
from pydantic import BaseModel, Field

from surface_conversion.translations import pre_refactor_logic
from utils.paths import ROOT_DIR


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


class ScanData(BaseModel, arbitrary_types_allowed=True):
    """For making testing more readable, dict is moved to pydantic model."""

    depth_data: NDArray[tuple[int, int]]
    xdim: float = Field(
        ...,
        description="x dimension of meters for 1 pixel",
        gt=0,
        le=1,
        examples=[0.7, 1],
    )
    ydim: float = Field(
        ...,
        description="y dimension of meters for 1 pixel",
        gt=0,
        le=1,
        examples=[0.7, 1],
    )


@pytest.fixture
def data_in(scan_image: ScanImage) -> ScanData:
    image = ScanImage.from_file(
        ROOT_DIR / "tests/resources/scans/Klein_non_replica_mode.al3d"
    )
    return ScanData(
        depth_data=image.data,
        xdim=image.scale_x,
        ydim=image.scale_y,
    )


def plot_test_data(data, show_plot=True) -> Figure:
    """Plot test data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")
    if show_plot:
        fig.show()
    return fig


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot_default"], extensions=["png"])
def test_get_surface_plot(data_in: ScanData) -> None:
    data = pre_refactor_logic(data_in.depth_data, data_in.xdim, data_in.ydim)
    plot_test_data(data, show_plot=False)
