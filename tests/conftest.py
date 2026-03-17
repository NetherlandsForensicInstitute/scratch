from collections.abc import Iterator
import pickle
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.likelihood_ratio import ModelSpecs
from conversion.plots.utils import build_results_metadata_impression
from conversion.profile_correlator import Profile
from fastapi.testclient import TestClient
from lir import LLRData

from constants import PROJECT_ROOT
from main import app
from models import DirectoryAccess
from settings import Settings
from tests.helper_function import (
    _create_dummy_profile,
    _save_striation_mark_and_profile,
    _shift_profile,
    _striation_mark,
)

TEST_ROOT = Path(__file__).parent
RESOURCES_DIR = TEST_ROOT / "resources"


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


@pytest.fixture(scope="session")
def mask_raw(mask: BinaryMask) -> bytes:
    return mask.tobytes(order="C")


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
def dummy_profile() -> Profile:
    return Profile(heights=np.random.default_rng(42).random(1000), pixel_size=1e-6)


@pytest.fixture
def mark_ref(dummy_mark: Mark) -> Mark:
    return dummy_mark


@pytest.fixture
def mark_comp(dummy_mark: Mark) -> Mark:
    return dummy_mark


@pytest.fixture
def striation_reference_data() -> ModelSpecs:
    return ModelSpecs(
        km_scores=np.array([0.9, 0.85, 0.78]),
        km_llrs=np.array([2.1, 1.8, 1.5]),
        km_llr_intervals=np.array([[1.9, 2.3], [1.6, 2.0], [1.3, 1.7]]),
        knm_scores=np.array([0.3, 0.25, 0.15, 0.1]),
        knm_llrs=np.array([-1.2, -0.9, -1.5, -2.0]),
        knm_llr_intervals=np.array([[-1.4, -1.0], [-1.1, -0.7], [-1.7, -1.3], [-2.2, -1.8]]),
    )


@pytest.fixture
def impression_reference_data() -> ModelSpecs:
    return ModelSpecs(
        km_scores=np.array([[0.0, 25, 30], [0.0, 22, 30], [0.0, 20, 30]]),
        km_llrs=np.array([2.1, 1.8, 1.5]),
        km_llr_intervals=np.array([[1.9, 2.3], [1.6, 2.0], [1.3, 1.7]]),
        knm_scores=np.array([[0.0, 5, 30], [0.0, 3, 30], [0.0, 2, 30], [0.0, 1, 30]]),
        knm_llrs=np.array([-1.2, -0.9, -1.5, -2.0]),
        knm_llr_intervals=np.array([[-1.4, -1.0], [-1.1, -0.7], [-1.7, -1.3], [-2.2, -1.8]]),
    )


@pytest.fixture
def results_metadata(impression_reference_data: ModelSpecs) -> dict[str, str]:
    return build_results_metadata_impression(
        reference_data=impression_reference_data,
        llr_data=LLRData(features=np.array([5.19])),
        date_report=date(2023, 2, 16),
        user_id="test_user",
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        score=4,
        n_cells=6,
        lr_system_path=Path("path/to/model"),
    )


@pytest.fixture
def mark_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Prepare directories with striation mark and profile files."""
    ref_path = tmp_path / "ref_mark"
    comp_path = tmp_path / "comp_mark"
    ref_path.mkdir()
    comp_path.mkdir()

    profile_ref = _create_dummy_profile()
    profile_comp = _shift_profile(profile_ref, 10.0)
    mark_ref = _striation_mark(profile_ref)
    mark_comp = _striation_mark(profile_comp)

    _save_striation_mark_and_profile(ref_path, profile_ref, mark_ref)
    _save_striation_mark_and_profile(comp_path, profile_comp, mark_comp)

    return ref_path, comp_path


@pytest.fixture(scope="session")
def impression_lr_system_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return RESOURCES_DIR / "impression"


@pytest.fixture(scope="session")
def striation_lr_system_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return RESOURCES_DIR / "striation"

def _create_dummy_impression_surface(rows: int = 100, cols: int = 100) -> np.ndarray:
    """Create a surface with enough variation for CMC to find cells."""
    rng = np.random.default_rng(42)
    # Base surface with some spatial structure
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    xs, ys = np.meshgrid(x, y)
    surface = np.sin(xs) * np.cos(ys) * 10.0
    # Add noise so cells have something to correlate
    surface += rng.normal(0, 1.0, surface.shape)
    return surface


@pytest.fixture
def impression_mark_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Prepare directories with impression mark files (processed + leveled)."""
    ref_path = tmp_path / "ref_mark"
    comp_path = tmp_path / "comp_mark"
    ref_path.mkdir()
    comp_path.mkdir()

    surface_ref = _create_dummy_impression_surface()
    # Shift slightly so marks are similar but not identical
    surface_comp = np.roll(surface_ref, shift=5, axis=1)

    mark_ref = _impression_mark(surface_ref)
    mark_comp = _impression_mark(surface_comp)

    # Both stems needed: ProcessedMark(filtered, leveled)
    for path, mark in [(ref_path, mark_ref), (comp_path, mark_comp)]:
        _save_impression_marks(path, mark)

    return ref_path, comp_path
