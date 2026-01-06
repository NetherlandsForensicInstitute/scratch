import pytest

from image_generation import compute_3d_image, get_array_for_display
from image_generation.data_formats import ScanImage

import numpy as np
from numpy.testing import assert_array_almost_equal


from ..constants import BASELINE_IMAGES_DIR


@pytest.mark.integration
def test_get_surface_plot(scan_image_replica: ScanImage) -> None:
    verified = np.load(BASELINE_IMAGES_DIR / "surface_plot.npy")
    generated_image = compute_3d_image(scan_image=scan_image_replica)
    assert np.array_equal(generated_image.data, verified)


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage,
):
    verified = np.load(BASELINE_IMAGES_DIR / "display_array.npy")
    display_image = get_array_for_display(scan_image_with_nans)
    assert_array_almost_equal(display_image.data, verified)
