import pickle
from pathlib import Path
from typing import Any, Protocol


class LRSystem(Protocol):
    """Protocol for objects that support sklearn-style prediction."""

    def predict(self, x: Any) -> Any:
        """Return predictions for input x."""
        ...


def get_lr_system(lr_system_path: Path) -> LRSystem:
    """Load an LR system (sklearn model) from a pickle file."""
    with lr_system_path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def calculate_lr(score: float, lr_system: LRSystem) -> float:
    """Calculate likelihood ratio by calling predict on the LR system."""
    return float(lr_system.predict([[score]])[0])
