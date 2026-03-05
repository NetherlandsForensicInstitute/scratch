import pickle
from pathlib import Path

import pytest
from computations.likelihood_ratio import (
    calculate_lr_impression,
    calculate_lr_striation,
    get_lr_system,
)
from lir.data.models import FeatureData, LLRData
from lir.lrsystems.lrsystems import LRSystem

RESOURCES = Path(__file__).parent.parent / "resources"


class _IdentityLRSystem(LRSystem):
    """Minimal LRSystem that returns the input score as the LLR."""

    def __init__(self) -> None:
        super().__init__(name="identity")

    def apply(self, instances: FeatureData) -> LLRData:
        """Return input features as LLR values."""
        return LLRData(features=instances.features)


@pytest.fixture
def identity_lr_system_path(tmp_path: Path) -> Path:
    """Pickle an identity LRSystem and return the path."""
    path = tmp_path / "lr_system.pkl"
    path.write_bytes(pickle.dumps(_IdentityLRSystem()))
    return path


@pytest.fixture
def random_lr_system_path() -> Path:
    """Path to the pre-built random LR system pickle in test resources."""
    return RESOURCES / "random_lr_system.pkl"


@pytest.mark.integration
class TestGetLRSystem:
    """Tests for get_lr_system."""

    def test_returns_object_with_apply(self, random_lr_system_path: Path) -> None:
        """Loaded object exposes an apply method."""
        model = get_lr_system(random_lr_system_path)
        assert hasattr(model, "apply")

    def test_returns_lrsystem(self, random_lr_system_path: Path) -> None:
        """Loaded object is a lir LRSystem."""
        model = get_lr_system(random_lr_system_path)
        assert isinstance(model, LRSystem)


@pytest.mark.integration
class TestCalculateLRStriation:
    """Tests for calculate_lr_striation."""

    def test_output_matches_model(self, identity_lr_system_path: Path) -> None:
        """Identity LRSystem returns the input score as the LLR."""
        model = get_lr_system(identity_lr_system_path)
        assert calculate_lr_striation(model, 0.5) == pytest.approx(0.5)

    def test_score_zero(self, identity_lr_system_path: Path) -> None:
        """Score of zero (no correlation) returns 0.0."""
        model = get_lr_system(identity_lr_system_path)
        assert calculate_lr_striation(model, 0.0) == pytest.approx(0.0)

    def test_score_negative(self, identity_lr_system_path: Path) -> None:
        """Negative correlation score (anti-correlation) returns the correct LLR."""
        model = get_lr_system(identity_lr_system_path)
        assert calculate_lr_striation(model, -0.5) == pytest.approx(-0.5)


@pytest.mark.integration
class TestCalculateLRImpression:
    """Tests for calculate_lr_impression."""

    def test_score_zero(self, identity_lr_system_path: Path) -> None:
        """Score of zero (no matching cells) returns 0.0."""
        model = get_lr_system(identity_lr_system_path)
        assert calculate_lr_impression(model, 0, 10) == pytest.approx(0.0)

    def test_score_negative(self, random_lr_system_path: Path) -> None:
        """Negative score does not raise."""
        model = get_lr_system(random_lr_system_path)
        assert isinstance(calculate_lr_impression(model, -1, 10), float)
