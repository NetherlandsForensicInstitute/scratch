from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage

from preprocessors.pipelines import parse_scan_pipeline

RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def parsed_al3d_file(scan_directory: Path) -> ScanImage:
    """Parse the circle.al3d test file."""
    return parse_scan_pipeline(scan_directory / "circle.al3d", 1, 1)


@pytest.fixture(scope="session")
def mask_original() -> BinaryMask:
    df = pd.read_csv(RESOURCES / "mask_original.csv.gz")
    return df.to_numpy(dtype=np.bool)


@pytest.fixture(scope="session")
def mask_bitpacked() -> bytes:
    df = pd.read_csv(RESOURCES / "mask_bitpacked.csv.gz")
    values = df.values.flatten().astype(np.uint8)
    return values.tobytes()
