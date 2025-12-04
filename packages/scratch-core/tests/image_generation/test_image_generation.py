import pytest
from matplotlib.testing.decorators import image_comparison

from image_generation import compute_3d_image
from image_generation.data_formats import ScanImage

from ..utils import plot_test_data


@pytest.fixture
def data_in(scan_image_replica: ScanImage) -> ScanImage:
    return scan_image_replica


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot_default"], extensions=["png"])
def test_get_surface_plot(data_in: ScanImage) -> None:
    generated_image = compute_3d_image(depth_data=data_in)
    plot_test_data(generated_image.data)
