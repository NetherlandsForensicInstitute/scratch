import pytest
from matplotlib.testing.decorators import image_comparison

from image_generation import generate_3d_image
from image_generation.data_formats import ScanMap2D
from parsers.data_types import ScanImage
from utils.paths import PROJECT_ROOT

from ..utils import plot_test_data  # type: ignore


@pytest.fixture
def data_in(scan_image: ScanImage) -> ScanImage:
    return ScanImage.from_file(
        PROJECT_ROOT / "tests/resources/scans/Klein_non_replica_mode.al3d"
    )


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot_default"], extensions=["png"])
def test_get_surface_plot(data_in: ScanImage) -> None:
    generated_image = generate_3d_image(
        depth_data=ScanMap2D(data=data_in.data),
        x_dimension=data_in.scale_x,
        y_dimension=data_in.scale_y,
    )
    plot_test_data(generated_image.data)
