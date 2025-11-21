import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.testing.decorators import image_comparison

from image_generation import get_surface_map
from image_generation.data_formats import Image2DArray
from parsers.data_types import ScanImage
from utils.paths import ROOT_DIR


def plot_test_data(data) -> Figure:
    """Plot test data for debugging purposes."""
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.axis("off")
    ax.axis("equal")
    return fig


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
    plot_test_data(data.data)
