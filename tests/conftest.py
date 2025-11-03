import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = SCRIPT_DIR / "resources"
SCANS_DIR = RESOURCES_DIR / "scans"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def image_data() -> np.ndarray:
    """Build a fixture with ground truth image data."""
    gray = Image.open(SCANS_DIR / "circle.png").convert("L")
    data = np.asarray(gray, dtype=np.float64)
    return data
