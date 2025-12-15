from math import ceil
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import micro
from surfalize import Surface

from container_models.scan_image import ScanImage
from parsers import load_scan_image
from returns.pipeline import is_successful

from parsers.loaders import subsample_array
from ..helper_function import unwrap_result


@pytest.fixture(scope="class")
def filepath(scans_dir: Path, request: pytest.FixtureRequest):
    return scans_dir / request.param


@pytest.mark.parametrize(
    "filepath",
    [
        "Klein_non_replica_mode.al3d",
        "Klein_non_replica_mode_X3P_Scratch.x3p",
    ],
    indirect=True,
)
class TestLoadScanImage:
    def test_load_scan_data_matches_size(self, filepath: Path) -> None:
        # Arrange
        surface = Surface.load(filepath)
        # Act
        result = load_scan_image(filepath)
        scan_image = unwrap_result(result)

        # Assert
        assert scan_image.data.shape == (
            ceil(surface.data.shape[0]),
            ceil(surface.data.shape[1]),
        )
        assert scan_image.scale_y == surface.step_y * micro
        assert scan_image.scale_x == surface.step_x * micro

    def test_load_scan_data_rejects_incorrect_sizes(
        self,
        filepath: Path,
    ) -> None:
        # act
        result = load_scan_image(filepath)
        # assert
        assert not is_successful(result)


# TODO: find a better test methology
def test_load_scan_data_matches_baseline_output(
    baseline_images_dir: Path, scans_dir: Path
) -> None:
    # arrange
    filepath = scans_dir / "Klein_non_replica_mode.al3d"
    verified = np.load(baseline_images_dir / "replica_subsampled.npy")
    # act
    result = load_scan_image(filepath, step_size_x=10, step_size_y=15)
    scan_image = unwrap_result(result)
    # assert
    assert np.allclose(scan_image.data, verified, equal_nan=True, atol=1.0e-5)


@pytest.mark.parametrize(
    ("step_x", "step_y"),
    [
        pytest.param(10, 10, id="default value"),
        pytest.param(1, 10, id="only step y"),
        pytest.param(10, 1, id="only x"),
        pytest.param(10, 5, id="different x and y"),
    ],
)
def test_subsample_updates_scan_image(
    scan_image: ScanImage, step_x: int, step_y: int
) -> None:
    # Act
    result = subsample_array(scan_image, (step_x, step_y))
    scan_image = unwrap_result(result)

    # Assert
    assert np.isclose(scan_image.scale_x, scan_image.scale_x * step_x, atol=1.0e-3)
    assert np.isclose(scan_image.scale_y, scan_image.scale_y * step_y, atol=1.0e-3)
