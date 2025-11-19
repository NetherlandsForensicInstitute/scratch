from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
from PIL import Image

from parsers.data_types import ScanImage

TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def scans_dir() -> Path:
    """Path to resources scan directory."""
    return TEST_ROOT / "resources" / "scans"


@pytest.fixture
def image_data(scans_dir: Path) -> NDArray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(scans_dir / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data


@pytest.fixture
def scan_image(image_data: NDArray) -> ScanImage:
    """Build a `ScanImage` object`."""
    return ScanImage(data=image_data, path_to_original_image=Path("some/path/file.x3p"))


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Dynamically parametrize tests that request al3d_file_path."""

    # Get the scans directory
    scans_dir = TEST_ROOT / "resources/scans"

    if "al3d_file_path" in metafunc.fixturenames:
        al3d_paths = tuple(scans_dir.glob("*.al3d"))
        metafunc.parametrize(
            "al3d_file_path", al3d_paths, ids=tuple(path.name for path in al3d_paths)
        )

    if "x3p_file_path" in metafunc.fixturenames:
        x3p_paths = tuple(scans_dir.glob("*.al3d"))
        metafunc.parametrize(
            "x3p_file_path", x3p_paths, ids=tuple(path.name for path in x3p_paths)
        )
