import pytest
from matplotlib.testing.decorators import image_comparison

from image_generation import compute_3d_image, get_array_for_display
from image_generation.data_formats import ScanImage

from ..utils import plot_test_data
import numpy as np
from numpy.testing import assert_array_almost_equal


from ..constants import BASELINE_IMAGES_DIR


@pytest.mark.integration
@image_comparison(baseline_images=["surfaceplot_default"], extensions=["png"])
def test_get_surface_plot(scan_image_replica: ScanImage) -> None:
    generated_image = compute_3d_image(scan_image=scan_image_replica)
    plot_test_data(generated_image.data)


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage,
):
    verified = np.load(BASELINE_IMAGES_DIR / "display_array.npy")
    display_image = get_array_for_display(scan_image_with_nans)
    assert_array_almost_equal(display_image.data, verified)
