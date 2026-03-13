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
from conversion.likelihood_ratio import ModelSpecs
from conversion.plots.utils import build_results_metadata_impression
from conversion.profile_correlator import Profile
from fastapi.testclient import TestClient
from lir import FeatureData, LLRData
from lir.config.base import ContextAwareDict
from lir.config.lrsystem_architectures import ParsedLRSystem, parse_lrsystem

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


def _to_context_aware(d: dict, context: list[str]) -> ContextAwareDict:
    """Recursively convert nested dicts to ContextAwareDict."""
    converted = {k: _to_context_aware(v, context + [str(k)]) if isinstance(v, dict) else v for k, v in d.items()}
    return ContextAwareDict(context, **converted)


def _build_lr_system(config: dict, output_dir: Path) -> ParsedLRSystem:
    cad = _to_context_aware(config, ["test"])
    return parse_lrsystem(cad, output_dir)


STRIATION_CONFIG = {
    "architecture": "specific_source",
    "modules": {
        "method": "bootstrap",
        "n_bootstraps": 400,
        "seed": 0,
        "points": "data",
        "steps": {
            "kde": {"method": "kde", "bandwidth": "silverman"},
            "elub": "elub_bounder",
        },
    },
}

IMPRESSION_CONFIG = {
    "architecture": "specific_source",
    "modules": {
        "steps": {
            "select": {
                "method": "lrmodule.helpers.select_marktype_cmc",
            },
            "transform": {
                "method": "lrmodule.helpers.transform_marktype_rel_cmc",
            },
            "bootstrap": {
                "method": "bootstrap",
                "n_bootstraps": 400,
                "seed": 0,
                "points": "data",
                "steps": {
                    "kde": {"method": "kde", "bandwidth": "silverman"},
                    "elub": "elub_bounder",
                },
            },
        },
    },
}


def _write_lr_system_dir(tmp_path: Path, config: dict, features: np.ndarray, labels: np.ndarray) -> Path:
    training_data = FeatureData(features=features, labels=labels)

    lr_system = _build_lr_system(config, tmp_path)
    lr_system.fit(training_data)

    with (tmp_path / "model.pkl").open("wb") as f:
        pickle.dump(lr_system, f)

    n_features = features.shape[1] if features.ndim > 1 else 1
    feature_cols = ",".join(f"feature_{i}" for i in range(n_features))
    fmt = ["%d"] + ["%.18e"] * n_features
    data = np.column_stack([labels, features])
    np.savetxt(
        tmp_path / "reference_data.csv",
        data,
        delimiter=",",
        header=f"hypothesis,{feature_cols}",
        comments="",
        fmt=fmt,
    )

    return tmp_path


@pytest.fixture
def striation_lr_system_path(tmp_path: Path) -> Path:
    """Directory with a fitted striation LR system and reference data."""
    rng = np.random.default_rng(42)
    km_features = np.clip(rng.normal(loc=0.8, scale=0.1, size=(50, 1)), -0.99, 0.99)
    knm_features = np.clip(rng.normal(loc=0.3, scale=0.1, size=(50, 1)), -0.99, 0.99)
    features = np.vstack([km_features, knm_features])
    labels = np.concatenate([np.ones(50), np.zeros(50)])
    return _write_lr_system_dir(tmp_path, STRIATION_CONFIG, features, labels)


@pytest.fixture
def impression_lr_system_path(tmp_path: Path) -> Path:
    """Directory with a fitted impression LR system and reference data."""
    rng = np.random.default_rng(42)
    n_cells = 30
    km_features = np.column_stack([
        np.zeros(50),  # dummy (column 0, unused)
        rng.integers(15, n_cells, size=50).astype(float),  # cmc (column 1)
        np.full(50, n_cells, dtype=float),  # n (column 2)
    ])
    knm_features = np.column_stack([
        np.zeros(50),
        rng.integers(0, 10, size=50).astype(float),
        np.full(50, n_cells, dtype=float),
    ])
    features = np.vstack([km_features, knm_features])
    labels = np.concatenate([np.ones(50), np.zeros(50)])
    return _write_lr_system_dir(tmp_path, IMPRESSION_CONFIG, features, labels)


@pytest.fixture
def lr_system_path(striation_lr_system_path: Path) -> Path:
    return striation_lr_system_path
