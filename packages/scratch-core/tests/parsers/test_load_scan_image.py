from math import ceil
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import micro
from surfalize import Surface

from parsers import load_scan_image
from returns.pipeline import is_successful

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
    @pytest.mark.parametrize("step_x, step_y", [(1, 1), (10, 10), (25, 25), (25, 50)])
    def test_load_scan_data_matches_size(
        self, filepath: Path, step_x: int, step_y: int
    ) -> None:
        # Arrange
        surface = Surface.load(filepath)
        # Act
        result = load_scan_image(filepath, step_x, step_y)
        scan_image = unwrap_result(result)

        # Assert
        assert scan_image.data.shape == (
            ceil(surface.data.shape[0] / step_y),
            ceil(surface.data.shape[1] / step_x),
        )

    @pytest.mark.parametrize(
        ("step_x", "step_y"),
        [
            pytest.param(10, 10, id="default value"),
            pytest.param(1, 10, id="only step y"),
            pytest.param(10, 1, id="only x"),
            pytest.param(10, 5, id="different x and y"),
        ],
    )
    def test_scan_map_updates_scales(
        self, filepath: Path, step_x: int, step_y: int
    ) -> None:
        # arrange
        surface = Surface.load(filepath)
        # Act
        result = load_scan_image(filepath, step_x, step_y)
        scan_image = unwrap_result(result)

        # Assert
        assert np.isclose(scan_image.scale_x, surface.step_x * step_x * micro)
        assert np.isclose(scan_image.scale_y, surface.step_y * step_y * micro)

    @pytest.mark.parametrize(
        "step_x, step_y", [(-2, 2), (0, 0), (0, 3), (2, -1), (-1, -1), (1e3, 1e4)]
    )
    def test_load_scan_data_rejects_incorrect_sizes(
        self, filepath: Path, step_x: int, step_y: int
    ) -> None:
        # act
        result = load_scan_image(filepath, step_x, step_y)
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
