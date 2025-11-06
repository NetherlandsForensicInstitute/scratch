from hashlib import sha256
from pathlib import Path

from lrmodule.data_types import ModelSettings
from lrmodule.lrsystem import ScratchLrSystem


def _get_model_dirname(settings: ModelSettings, dataset_id: str) -> str:
    h = sha256()
    h.update(str(settings).encode("utf8"))
    h.update(dataset_id.encode("utf8"))
    return h.hexdigest()


def load_model(settings: ModelSettings, dataset_id: str, cache_dir: Path) -> ScratchLrSystem | None:
    """Load previously cached model."""
    cache_dir / _get_model_dirname(settings, dataset_id) / "model.pkl"
    raise NotImplementedError


def save_model(model: ScratchLrSystem, cache_dir: Path) -> None:
    """Save a model to disk."""
    cache_dir / _get_model_dirname(model.settings, model.dataset_id)
    raise NotImplementedError
