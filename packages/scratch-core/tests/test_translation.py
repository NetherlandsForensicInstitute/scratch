import numpy as np
import pytest
from _translation import get_surface_plot
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison
from numpy._typing import NDArray
from parsers.data_types import ScanImage


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


@pytest.fixture
def data_in(scan_image: ScanImage) -> dict[str, np.ndarray]:
    return {
        "depth_data": scan_image.data,
        "xdim": scan_image.width,
        "ydim": scan_image.height,
    }


def plot_test_data(data) -> Figure:
    """Plot test data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")
    return fig


def test_get_surface_plot(data_in):
    data = get_surface_plot(data_in)
    plot_test_data(data).show()


def test_get_surface_plot_data_with_extra_light(data_in):
    data = get_surface_plot(data_in, True)
    plot_test_data(data).show()


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
def test_get_surface_plot_data_with_light_angle(data_in: dict, light_angles: tuple[list[int]]):
    data = get_surface_plot(data_in, True, np.array(light_angles))
    plot_test_data(data).show()


def test_get_surface_plot_data_with_mask(data_in, mask):
    data = get_surface_plot(data_in, True, np.array([[90, 45], [180, 45]]), mask)
    plot_test_data(data).show()


def test_get_surface_plot_data_with_cropping_mask(data_in, mask_crop):
    data = get_surface_plot(data_in, True, np.array([[90, 45], [180, 45]]), mask_crop)
    plot_test_data(data).show()


def test_get_surface_plot_data_with_mask_and_zoom(data_in, mask_crop):
    data = get_surface_plot(data_in, True, np.array([[90, 45], [180, 45]]), mask_crop, True)
    plot_test_data(data).show()


def test_get_surface_plot_data_with_mask_and_zoom_failed(data_in, mask):
    data = get_surface_plot(data_in, True, np.array([[90, 45], [180, 45]]), mask, True)
    plot_test_data(data).show()


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot"], extensions=["png"])
def test_get_surface_plot_integration(data_in, mask_crop):
    """This integration test will loop through all options of the get_surface_plot."""
    data = get_surface_plot(data_in, True, np.array([[90, 45], [180, 45]]), mask_crop, True)
    plot_test_data(data)
