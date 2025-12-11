from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from renders import get_array_for_display
from container_models.scan_image import ScanImage


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage, baseline_images_dir: Path
):
    # arrange
    verified = np.load(baseline_images_dir / "display_array.npy")
    # act
    display_image = get_array_for_display(scan_image_with_nans).unwrap()
    # assert
    assert_array_almost_equal(display_image.data, verified)
