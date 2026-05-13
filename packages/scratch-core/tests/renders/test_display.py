from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL.Image import fromarray, open
from preprocessors.pipelines import preview_pipeline

from container_models.scan_image import ScanImage


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage, baseline_images_dir: Path, tmp_path: Path
):
    # arrange
    verified = np.load(baseline_images_dir / "display_array.npy")
    captured = {}
    output_file = tmp_path / "image.png"

    original_fromarray = fromarray

    def fake_fromarray(obj, *args, **kwargs):
        captured["array"] = obj.copy()
        return original_fromarray(obj, *args, **kwargs)

    # act
    # TODO: break up preview_pipeline into separate steps to test.
    with patch(
        "container_models.scan_image.fromarray",
        new=fake_fromarray,
    ):
        preview_pipeline(scan_image_with_nans, output_path=output_file)
    # assert
    assert "array" in captured, "mock should have saved array in captured"
    assert np.allclose(
        verified,
        captured["array"],
        equal_nan=True,
    ), "numpy images should be the same before the save to png."
    assert output_file.exists()
    with open(output_file) as img:
        img.verify()
