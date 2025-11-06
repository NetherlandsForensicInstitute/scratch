import numpy as np
import pytest
from _translation import get_surface_plot
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
from parsers.data_types import ScanImage


@pytest.fixture
def mask(scan_image: ScanImage):
    mask = np.zeros((scan_image.width, scan_image.height), dtype=np.uint8)
    square_size = 40
    top_left = (100, 150)
    mask[top_left[0] : top_left[0] + square_size, top_left[1] : top_left[1] + square_size] = 1
    return mask


@pytest.fixture
def data_in(scan_image: ScanImage) -> dict[str, np.ndarray]:
    return {
        "depth_data": scan_image.data,
        "xdim": scan_image.width,
        "ydim": scan_image.height,
    }


def plot_test_data(data) -> None:
    """Plot test data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")


def test_get_surface_plot(data_in):
    data = get_surface_plot(data_in)
    plot_test_data(data)


def test_get_surface_plot_data_with_extra_light(data_in):
    data = get_surface_plot(data_in, True)
    plot_test_data(data)


@pytest.mark.parametrize(
    "light_angles",
    [
        ([90, 45]),
        ([90, 45], [180, 45]),
        ([90, 45], [180, 45], [270, 90]),
        ([90, 45], [180, 45], [270, 90], [90, 90]),
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
    plot_test_data(data)


def test_get_surface_plot_data_with_mask(data_in, mask):
    data = get_surface_plot(data_in, True, np.array([[90, 45], [180, 45]]), mask)
    plot_test_data(data)


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot"], extensions=["png"])
def test_get_surface_plot_integration(data_in, mask):
    data = get_surface_plot(data_in, True, np.array([[90, 45], [180, 45]]), mask)
    plot_test_data(data)
