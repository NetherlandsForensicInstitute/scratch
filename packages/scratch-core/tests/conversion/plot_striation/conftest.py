import numpy as np
import pytest

from conversion.data_formats import Mark
from conversion.plots.data_formats import CorrelationMetrics
from .helper_functions import (
    create_synthetic_striation_data,
    create_mock_mark,
    create_mock_profile_mark,
)


@pytest.fixture
def profile_ref() -> np.ndarray:
    return create_synthetic_striation_data(height=1, width=200, seed=42)


@pytest.fixture
def profile_comp() -> np.ndarray:
    return create_synthetic_striation_data(height=1, width=200, seed=43)


@pytest.fixture
def surface_ref() -> np.ndarray:
    return create_synthetic_striation_data(height=256, width=200, seed=42)


@pytest.fixture
def surface_comp() -> np.ndarray:
    return create_synthetic_striation_data(height=256, width=220, seed=43)


@pytest.fixture
def mark_ref() -> Mark:
    return create_mock_mark(height=256, width=200, seed=42)


@pytest.fixture
def mark_comp() -> Mark:
    return create_mock_mark(height=256, width=220, seed=43)


@pytest.fixture
def mark_ref_aligned() -> Mark:
    return create_mock_mark(height=200, width=200, seed=44)


@pytest.fixture
def mark_comp_aligned() -> Mark:
    return create_mock_mark(height=200, width=200, seed=45)


@pytest.fixture
def profile_mark_ref() -> Mark:
    return create_mock_profile_mark(length=200, seed=46)


@pytest.fixture
def profile_mark_comp() -> Mark:
    return create_mock_profile_mark(length=200, seed=47)


@pytest.fixture
def quality_passbands() -> dict[tuple[float, float], float]:
    return {
        (5, 250): 0.85,
        (100, 250): 0.78,
        (50, 100): 0.65,
        (25, 50): 0.45,
        (10, 25): 0.30,
        (5, 10): 0.15,
    }


@pytest.fixture
def metrics(quality_passbands) -> CorrelationMetrics:
    return CorrelationMetrics(
        score=0.85,
        shift=12.5,
        overlap=80.4,
        sq_a=0.2395,
        sq_b=0.7121,
        sq_b_minus_a=0.6138,
        sq_ratio=297.3765,
        sign_diff_dsab=220.94,
        data_spacing=1.5625,
        quality_passbands=quality_passbands,
    )


@pytest.fixture
def metadata_ref() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_1",
        "Measurement ID": "striated_mark",
    }


@pytest.fixture
def metadata_comp() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_2",
        "Measurement ID": "striated_mark",
    }
