import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from conversion.display import get_array_for_display
from image_generation.data_formats import ScanImage

from ..constants import BASELINE_IMAGES_DIR


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage,
):
    verified = np.load(BASELINE_IMAGES_DIR / "display_array.npy")
    display_image = get_array_for_display(scan_image_with_nans)
    assert_array_almost_equal(display_image, verified)
