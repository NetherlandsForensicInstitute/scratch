import pickle
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.likelihood_ratio import ReferenceData
from conversion.plots.data_formats import ImpressionComparisonMetrics
from conversion.plots.utils import build_results_metadata_impression
from fastapi.testclient import TestClient
from lir import LLRData

from constants import PROJECT_ROOT
from main import app
from models import DirectoryAccess
from settings import Settings
from tests.processors.conftest import _IdentityLRSystem


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


@pytest.fixture
def lr_system_path(tmp_path: Path) -> Path:
    """Pickle an identity LRSystem and return its path."""
    path = tmp_path / "lr_system.pkl"
    path.write_bytes(pickle.dumps(_IdentityLRSystem()))
    return path


@pytest.fixture
def mark_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create empty mark directories."""
    ref = tmp_path / "mark_ref"
    comp = tmp_path / "mark_comp"
    ref.mkdir()
    comp.mkdir()
    return ref, comp


@pytest.fixture
def dummy_mark() -> Mark:
    return Mark(
        scan_image=ScanImage(
            data=np.random.default_rng(42).random((50, 50)).astype(np.float64),
            scale_x=1e-6,
            scale_y=1e-6,
        ),
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
    )


@pytest.fixture
def mark_ref(dummy_mark: Mark) -> Mark:
    return dummy_mark


@pytest.fixture
def mark_comp(dummy_mark: Mark) -> Mark:
    return dummy_mark


@pytest.fixture
def reference_data() -> ReferenceData:
    return ReferenceData(
        km_model="random",
        km_scores=np.array([0.9, 0.85, 0.78]),
        km_llr_data=LLRData(
            features=np.array([
                [2.1, 1.9, 2.3],
                [1.8, 1.6, 2.0],
                [1.5, 1.3, 1.7],
            ])
        ),
        knm_model="random",
        knm_scores=np.array([0.3, 0.25, 0.15, 0.1]),
        knm_llr_data=LLRData(
            features=np.array([
                [-1.2, -1.4, -1.0],
                [-0.9, -1.1, -0.7],
                [-1.5, -1.7, -1.3],
                [-2.0, -2.2, -1.8],
            ])
        ),
    )


@pytest.fixture
def impression_metrics() -> ImpressionComparisonMetrics:
    return ImpressionComparisonMetrics(
        area_correlation=0.82,
        cell_correlations=np.array([[0.75, 0.88], [0.92, 0.31]]),
        cmc_score=66.7,
        mean_square_ref=1.25,
        mean_square_comp=1.31,
        mean_square_of_difference=0.42,
        has_area_results=True,
        has_cell_results=True,
        cell_positions_compared=np.array([[10.0, 20.0], [10.0, 60.0], [50.0, 20.0], [50.0, 60.0]]),
        cell_rotations_compared=np.array([0.01, -0.02, 0.0, -0.01]),
        cmc_area_fraction=55.0,
        cutoff_low_pass=250.0,
        cutoff_high_pass=25.0,
        cell_size_um=300.0,
        max_error_cell_position=50.0,
        max_error_cell_angle=3.0,
    )


@pytest.fixture
def results_metadata(reference_data: ReferenceData) -> dict[str, str]:
    return build_results_metadata_impression(
        reference_data=reference_data,
        llr_data=LLRData(features=np.array([5.19])),
        date_report=date(2023, 2, 16),
        user_id="test_user",
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        score=4,
        n_cells=6,
    )


@pytest.fixture
def mark_dir_ref(tmp_path: Path) -> Path:
    directory = tmp_path / "mark_ref"
    directory.mkdir()
    return directory


@pytest.fixture
def mark_dir_comp(tmp_path: Path) -> Path:
    directory = tmp_path / "mark_comp"
    directory.mkdir()
    return directory


@pytest.fixture
def lr_system_file(tmp_path: Path) -> Path:
    f = tmp_path / "lr_system.bin"
    f.touch()
    return f
