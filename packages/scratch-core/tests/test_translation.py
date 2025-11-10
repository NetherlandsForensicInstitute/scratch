import numpy as np
import pytest
from numpy.testing import assert_allclose

from _translation import get_surface_plot, convert_image_to_slope_map
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


@image_comparison(baseline_images=["surfaceplot_default"], extensions=["png"])
def test_get_surface_plot(data_in: ScanData) -> None:
    data = get_surface_plot(data_in.model_dump())
    plot_test_data(data, show_plot=False)


@image_comparison(baseline_images=["surfaceplot_extra_light"], extensions=["png"])
def test_get_surface_plot_data_with_extra_light(data_in: ScanData) -> None:
    data = get_surface_plot(data_in.model_dump(), True)
    plot_test_data(data, show_plot=False)


@pytest.mark.parametrize(
    "light_angles",
    [
        # ([90, 45]),
        ([90, 45], [180, 45]),
        ([90, 45], [180, 45], [270, 45]),
        ([90, 45], [180, 45], [270, 45], [90, 90]),
    ],
    ids=[
        # "one light source", not working yet
        "default value",
        "same as extra light source",
        "extra lights",
    ],
)
def test_get_surface_plot_data_with_light_angle(
    data_in: ScanData, light_angles: tuple[list[int]]
) -> None:
    data = get_surface_plot(data_in.model_dump(), None, np.array(light_angles))
    plot_test_data(data)


@image_comparison(baseline_images=["surfaceplot_with_masking"], extensions=["png"])
def test_get_surface_plot_data_with_mask(
    data_in: ScanData, mask: NDArray[tuple[int, int]]
) -> None:
    data = get_surface_plot(data_in.model_dump(), None, np.array([]), mask)
    plot_test_data(data, show_plot=False)


@image_comparison(
    baseline_images=["surfaceplot_with_cropping_mask"], extensions=["png"]
)
def test_get_surface_plot_data_with_cropping_mask(
    data_in: ScanData, mask_crop: NDArray[tuple[int, int]]
) -> None:
    data = get_surface_plot(data_in.model_dump(), None, np.array([]), mask_crop)
    plot_test_data(data, show_plot=False)


@image_comparison(baseline_images=["surfaceplot_mask_and_zoom"], extensions=["png"])
def test_get_surface_plot_data_with_mask_and_zoom(
    data_in: ScanData, mask_crop: NDArray[tuple[int, int]]
) -> None:
    data = get_surface_plot(data_in.model_dump(), None, np.array([]), mask_crop, True)
    plot_test_data(data, show_plot=False)


@image_comparison(baseline_images=["surfaceplot_zoom_failed"], extensions=["png"])
def test_get_surface_plot_data_with_mask_and_zoom_failed(
    data_in: ScanData, mask: NDArray[tuple[int, int]]
) -> None:
    data = get_surface_plot(data_in.model_dump(), None, np.array([]), mask, True)
    plot_test_data(data, show_plot=False)


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot"], extensions=["png"])
def test_get_surface_plot_integration(
    data_in: ScanData, mask_crop: NDArray[tuple[int, int]]
) -> None:
    """This integration test will provide all arguments of the get_surface_plot."""
    data = get_surface_plot(
        data_in.model_dump(), True, np.array([[90, 45], [30, 30]]), mask_crop, True
    )
    plot_test_data(data, show_plot=False)


class TestSurfaceSlopeConversion:
    # TODO: maybe move the outer border is NaN test to own test and only test the slopes
    def test_flat_surface_returns_upward_normal(self):
        """Given a flat surface the depth map should also be flat.
        The image is 1 pixel smaller on all sides due to the slope calculation.
        This is filled with NaN values to get the same shape as original image
        """
        # Arrange
        input_image = np.zeros((4, 4))
        inner_mask = np.zeros_like(input_image, dtype=bool)
        inner_mask[:-1, :-1] = True
        outer_mask = ~inner_mask

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, 1, 1)

        # Assert
        assert n1.shape == input_image.shape
        assert_allclose(n1[inner_mask], 0), "innerside should be 0 (no x direction)"
        assert_allclose(n2[inner_mask], 0), "innerside should be 0 (no y direction)"
        assert_allclose(n3[inner_mask], 1), "innerside should be 1 (no z direction)"
        assert np.any(n1[outer_mask]), "outer row and columns should be NaN"
        assert np.any(n2[outer_mask]), "outer row and columns should be NaN"
        assert np.any(n3[outer_mask]), "outer row and columns should be NaN"

    def test_linear_slope_in_y_direction(self):
        """Test the conversion if the immage has a slope of 2 to the right."""
        # Arrange
        max_number = 20
        step_y = 2
        step_x = 0
        norm = np.sqrt(step_x**2 + step_y**2 + 1)

        num_steps = int((step_y / max_number) + 1)
        input_image = np.tile(np.linspace(0, max_number, num_steps), (4, 1))
        inner_mask = np.zeros_like(input_image, dtype=bool)
        inner_mask[:-1, :-1] = True

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, xdim=1, ydim=1)

        # Assertion
        expected_n1 = -step_x / norm  # x-component
        expected_n2 = -step_y / norm  # y-component
        expected_n3 = 1 / norm  # z-component

        assert_allclose(n1[inner_mask], expected_n1, atol=1e-6)
        assert_allclose(n2[inner_mask], expected_n2, atol=1e-6)
        assert_allclose(n3[inner_mask], expected_n3, atol=1e-6)
