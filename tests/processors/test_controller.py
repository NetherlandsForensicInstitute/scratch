import pickle
from pathlib import Path

import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from processors.controller import calculate_lr, get_lr_system

TOLERANCE = 1e-6


@pytest.fixture
def dummy_lr_system_pickle(tmp_path: Path) -> Path:
    """Train a LinearRegression on y=x and write it to a temp pickle file."""
    model = LinearRegression()
    model.fit([[0.0], [1.0]], [0.0, 1.0])
    path = tmp_path / "lr_system.pkl"
    path.write_bytes(pickle.dumps(model))
    return path


@pytest.mark.integration
class TestGetLRSystem:
    """Tests for get_lr_system."""

    def test_returns_object_with_predict(self, dummy_lr_system_pickle: Path) -> None:
        """Loaded object exposes a predict method."""
        model = get_lr_system(dummy_lr_system_pickle)
        assert hasattr(model, "predict")

    def test_returns_sklearn_estimator(self, dummy_lr_system_pickle: Path) -> None:
        """Loaded object is an sklearn BaseEstimator."""
        model = get_lr_system(dummy_lr_system_pickle)
        assert isinstance(model, BaseEstimator)


class TestCalculateLR:
    """Tests for calculate_lr."""

    def test_returns_float(self, dummy_lr_system_pickle: Path) -> None:
        """Return type is float."""
        model = get_lr_system(dummy_lr_system_pickle)
        assert isinstance(calculate_lr(0.5, model), float)

    def test_predict_output_matches_model(self, dummy_lr_system_pickle: Path) -> None:
        """LinearRegression trained on y=x predicts ~0.5 for score=0.5."""
        model = get_lr_system(dummy_lr_system_pickle)
        assert abs(calculate_lr(0.5, model) - 0.5) < TOLERANCE
