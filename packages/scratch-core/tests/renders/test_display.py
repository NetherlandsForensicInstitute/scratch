from pathlib import Path
import numpy as np
import pytest

from renders import get_scan_image_for_display
from container_models import ImageContainer


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ImageContainer, baseline_images_dir: Path
):
    # arrange
    verified = np.load(baseline_images_dir / "display_array.npy")
    # act
    display_image = get_scan_image_for_display(scan_image_with_nans).unwrap()
    # assert
    assert np.allclose(verified, display_image.data, equal_nan=True)
