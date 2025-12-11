from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from constants import PROJECT_ROOT
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def scan_directory() -> Path:
    return PROJECT_ROOT / "packages/scratch-core/tests/resources/scans"
