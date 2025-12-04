import pytest
from matplotlib.testing.decorators import image_comparison

from image_generation import compute_3d_image
from image_generation.data_formats import ScanImage
from parsers.data_types import load_scan_image
from utils.paths import PROJECT_ROOT

from ..utils import plot_test_data  # type: ignore


@pytest.fixture
def data_in(scan_image: ScanImage) -> ScanImage:
    return load_scan_image(
        PROJECT_ROOT / "tests/resources/scans/Klein_non_replica_mode.al3d"
    )


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot_default"], extensions=["png"])
def test_get_surface_plot(data_in: ScanImage) -> None:
    generated_image = compute_3d_image(depth_data=data_in)
    plot_test_data(generated_image.data)
