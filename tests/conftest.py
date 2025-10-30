import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = SCRIPT_DIR / "resources"
SCANS_DIR = RESOURCES_DIR / "scans"


@pytest.fixture
def image_data() -> np.ndarray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(SCANS_DIR / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data
