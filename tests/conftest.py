import os

os.environ.setdefault("MPLBACKEND", "Agg")

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from container_models.base import BinaryMask
from fastapi.testclient import TestClient

from constants import PROJECT_ROOT
from main import app
from models import DirectoryAccess
from settings import Settings


@pytest.fixture(scope="session")
def tmp_dir_api(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Configure DirectoryAccess to use a temporary directory via settings."""
    # Create a temporary directory for testing
    temp_dir = tmp_path_factory.mktemp("temp_dir_api")

    with patch(
        "settings.get_settings",
        return_value=Settings.model_construct(STORAGE=temp_dir),
    ):
        yield


@pytest.fixture(scope="module")
def directory_access() -> DirectoryAccess:
    directory = DirectoryAccess(tag="test")
    directory.resource_path.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def scan_directory() -> Path:
    return PROJECT_ROOT / "packages/scratch-core/tests/resources/scans"


@pytest.fixture(scope="session")
def mask() -> BinaryMask:
    array = np.zeros(shape=(259, 259), dtype=np.bool_)
    array[1:259, 1:259] = True
    return array
