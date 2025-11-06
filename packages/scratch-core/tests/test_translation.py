import numpy as np
import pytest
from _translation import get_surface_plot
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison
from numpy._typing import NDArray
from parsers.data_types import ScanImage
from pydantic import BaseModel


@pytest.fixture
def mask_crop(scan_image: ScanImage) -> NDArray[tuple[int, int]]:
    """Makes a mask used for cropping (all is 0 except the drawn square)."""
    mask = np.zeros((scan_image.width, scan_image.height), dtype=np.uint8)
    square_size = 85
    x_start_cor = 100
    y_start_cor = 150
    mask[x_start_cor : x_start_cor + square_size, y_start_cor : y_start_cor + square_size] = 1
    return mask


@pytest.fixture
def mask(scan_image: ScanImage) -> NDArray[tuple[int, int]]:
    """Makes a mask used for masking that area (all is 1 except the drawn square)."""
    mask = np.ones((scan_image.width, scan_image.height), dtype=np.uint8)
    square_size = 85
    x_start_cor = 100
    y_start_cor = 150
    mask[x_start_cor : x_start_cor + square_size, y_start_cor : y_start_cor + square_size] = 0
    return mask


class ScanData(BaseModel, arbitrary_types_allowed=True):
    """For making testing more readable, dict is moved to pydantic model."""

    depth_data: NDArray[tuple[int, int]]
    xdim: int
    ydim: int


@pytest.fixture
def data_in(scan_image: ScanImage) -> ScanData:
    return ScanData(
        depth_data=scan_image.data,
        xdim=scan_image.width,
        ydim=scan_image.height,
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


def test_get_surface_plot(data_in: ScanData) -> None:
    data = get_surface_plot(data_in.model_dump())
    plot_test_data(data)


def test_get_surface_plot_data_with_extra_light(data_in: ScanData) -> None:
    data = get_surface_plot(data_in.model_dump(), True)
    plot_test_data(data)


@pytest.mark.parametrize(
    "light_angles",
    [
        ([90, 45]),
        ([90, 45], [180, 45]),
        ([90, 45], [180, 45], [270, 45]),
        ([90, 45], [180, 45], [270, 45], [90, 90]),
    ],
    ids=[
        "one light source",
        "default value",
        "same as extra light source",
        "extra lights",
    ],
)
def test_get_surface_plot_data_with_light_angle(data_in: ScanData, light_angles: tuple[list[int]]) -> None:
    data = get_surface_plot(data_in.model_dump(), True, np.array(light_angles))
    plot_test_data(data)


def test_get_surface_plot_data_with_mask(data_in: ScanData, mask: NDArray[tuple[int, int]]) -> None:
    light_angles = np.array([[90, 45], [180, 45]])
    data = get_surface_plot(data_in.model_dump(), True, light_angles, mask)
    plot_test_data(data)


def test_get_surface_plot_data_with_cropping_mask(data_in: ScanData, mask_crop: NDArray[tuple[int, int]]) -> None:
    data = get_surface_plot(data_in.model_dump(), True, np.array([[90, 45], [180, 45]]), mask_crop)
    plot_test_data(data)


def test_get_surface_plot_data_with_mask_and_zoom(data_in: ScanData, mask_crop: NDArray[tuple[int, int]]) -> None:
    data = get_surface_plot(data_in.model_dump(), True, np.array([[90, 45], [180, 45]]), mask_crop, True)
    plot_test_data(data)


def test_get_surface_plot_data_with_mask_and_zoom_failed(data_in: ScanData, mask: NDArray[tuple[int, int]]) -> None:
    data = get_surface_plot(data_in.model_dump(), True, np.array([[90, 45], [180, 45]]), mask, True)
    plot_test_data(data)


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot"], extensions=["png"])
def test_get_surface_plot_integration(data_in: ScanData, mask_crop: NDArray[tuple[int, int]]) -> None:
    """This integration test will provide all arguments of the get_surface_plot."""
    data = get_surface_plot(data_in.model_dump(), True, np.array([[90, 45], [30, 30]]), mask_crop, True)
    plot_test_data(data, show_plot=False)
