import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from dependencies import get_tmp_dir, get_token
from main import app


@pytest.fixture
def tmp_dir_api(tmp_path: Path) -> Iterator[Path]:
    """Replace get_temp_dir to return tmp_path."""
    tmp_dir = tmp_path / uuid4().hex
    tmp_dir.mkdir()
    app.dependency_overrides[get_tmp_dir] = lambda: tmp_dir
    yield tmp_dir
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    app.dependency_overrides.clear()


@pytest.fixture
def token() -> Iterator[UUID]:
    """Fixture that provides a fixed token for testing using dependency override."""
    fixed_token = uuid4()
    app.dependency_overrides[get_token] = lambda: fixed_token
    yield fixed_token
    app.dependency_overrides.clear()
