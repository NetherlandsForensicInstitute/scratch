import numpy as np
import pytest
from numpy.testing import assert_allclose

from _translation import get_surface_plot, convert_image_to_slope_map
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison
from numpy._typing import NDArray
from parsers.data_types import ScanImage
from pydantic import BaseModel, Field

from utils import ROOT_DIR


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
    image = ScanImage.from_file(ROOT_DIR / "tests/resources/scans/Huls1.al3d")
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
    plot_test_data(data, show_plot=False)


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
    TEST_IMAGE_WIDTH = 20
    TEST_IMAGE_HEIGHT = 20
    TOLERANCE = 1e-6

    @pytest.fixture(scope="class")
    def inner_mask(self) -> NDArray[tuple[int, int]]:
        inner_mask = np.zeros(
            (self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT), dtype=bool
        )
        inner_mask[:-1, :-1] = True
        return inner_mask

    def test_slope_has_nan_border(self, inner_mask: NDArray[tuple[int, int]]) -> None:
        """The image is 1 pixel smaller on all sides due to the slope calculation.
        This is filled with NaN values to get the same shape as original image"""
        # Arrange
        input_image = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))
        outer_mask = ~inner_mask

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, 1, 1)

        # Assert
        assert not np.any(np.isnan(n1[inner_mask])), (
            "inner row and columns should have a number"
        )
        assert not np.any(np.isnan(n2[inner_mask])), (
            "outer row and columns should have a number"
        )
        assert not np.any(np.isnan(n3[inner_mask])), (
            "outer row and columns should have a number"
        )
        assert np.any(np.isnan(n1[outer_mask])), "outer row and columns should be NaN"
        assert np.any(np.isnan(n2[outer_mask])), "outer row and columns should be NaN"
        assert np.any(np.isnan(n3[outer_mask])), "outer row and columns should be NaN"

    def test_flat_surface_returns_upward_normal(
        self, inner_mask: NDArray[tuple[int, int]]
    ) -> None:
        """Given a flat surface the depth map should also be flat."""
        # Arrange
        input_image = np.zeros((self.TEST_IMAGE_WIDTH, self.TEST_IMAGE_HEIGHT))

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, 1, 1)

        # Assert
        assert n1.shape == input_image.shape
        assert_allclose(n1[inner_mask], 0), "innerside should be 0 (no x direction)"
        assert_allclose(n2[inner_mask], 0), "innerside should be 0 (no y direction)"
        assert_allclose(n3[inner_mask], 1), "innerside should be 1 (no z direction)"

    @pytest.mark.parametrize(
        "step_x, step_y",
        [
            (2, 0),
            (0, 2),
            (2, 2),
        ],
        ids=["only increase in x", "only increase in y", "increase in x and y"],
    )
    def test_linear_slope(self, step_x, step_y, inner_mask: np.ndarray):
        """Test linear slopes in X, Y, or both directions."""
        # Arrange
        norm = np.sqrt(step_x**2 + step_y**2 + 1)
        expected_n1 = -step_x / norm
        expected_n2 = -step_y / norm
        expected_n3 = 1 / norm
        x_vals = np.arange(self.TEST_IMAGE_WIDTH) * step_x
        y_vals = np.arange(self.TEST_IMAGE_HEIGHT) * step_y
        input_image = y_vals[:, None] + x_vals[None, :]

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_image, xdim=1, ydim=1)

        # Assert
        (
            assert_allclose(n1[inner_mask], expected_n1, atol=self.TOLERANCE),
            (f"expected continuous n1 slope of {expected_n1}"),
        )
        (
            assert_allclose(n2[inner_mask], expected_n2, atol=self.TOLERANCE),
            (f"expected continuous n2 slope of {expected_n2}"),
        )
        (
            assert_allclose(n3[inner_mask], expected_n3, atol=self.TOLERANCE),
            (f"expected continuous n3 slope of {expected_n3}"),
        )

    def test_local_slope_location(self, inner_mask: NDArray[tuple[int, int]]) -> None:
        """Check that slope calculation is localized to the bump coordinates."""
        # Arrange
        image_size = self.TEST_IMAGE_WIDTH
        center_row = image_size // 2
        center_col = image_size // 2
        bumb_size = 4
        nan_offset = 1
        input_depth_map = np.zeros((image_size, image_size))

        bump_height = 6
        bump_rows = slice(center_row - bumb_size // 2, center_row + bumb_size // 2)
        bump_cols = slice(center_col - bumb_size // 2, center_col + bumb_size // 2)
        input_depth_map[bump_rows, bump_cols] = bump_height

        bump_mask = np.zeros_like(input_depth_map, dtype=bool)
        bump_mask[
            center_col - bumb_size // 2 - nan_offset : center_col + bumb_size // 2,
            center_row - bumb_size // 2 - nan_offset : center_row + bumb_size // 2,
        ] = True
        outside_bump_mask = ~bump_mask & inner_mask

        # Act
        n1, n2, n3 = convert_image_to_slope_map(input_depth_map, xdim=1, ydim=1)

        # Assert
        assert np.any(np.abs(n1[bump_mask]) > 0), "n1 should have slope inside bump"
        assert np.any(np.abs(n2[bump_mask]) > 0), "n2 should have slope inside bump"
        assert np.any(np.abs(n3[bump_mask]) != 1), (
            "n3 should deviate from 1 inside bump"
        )
        (
            assert_allclose(n1[outside_bump_mask], 0, atol=self.TOLERANCE),
            "outside the bumb X should be 0",
        )
        (
            assert_allclose(n2[outside_bump_mask], 0, atol=self.TOLERANCE),
            "outside the bumb Y should be 0",
        )
        (
            assert_allclose(n3[outside_bump_mask], 1, atol=self.TOLERANCE),
            "outside the bumb Z should be 1",
        )
