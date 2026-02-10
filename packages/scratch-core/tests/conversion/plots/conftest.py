import numpy as np
import pytest

from container_models.base import FloatArray2D
from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    StriationComparisonMetrics,
    ImpressionComparisonMetrics,
)

from .helper_functions import (
    create_synthetic_impression_data,
    create_synthetic_impression_mark,
    create_synthetic_profile_mark,
    create_synthetic_striation_data,
    create_synthetic_striation_mark,
)


@pytest.fixture
def striation_profile_reference() -> FloatArray2D:
    return create_synthetic_striation_data(height=1, width=200, seed=42)


@pytest.fixture
def striation_profile_compared() -> FloatArray2D:
    return create_synthetic_striation_data(height=1, width=200, seed=43)


@pytest.fixture
def striation_surface_reference() -> FloatArray2D:
    return create_synthetic_striation_data(height=256, width=200, seed=42)


@pytest.fixture
def striation_surface_compared() -> FloatArray2D:
    return create_synthetic_striation_data(height=256, width=220, seed=43)


@pytest.fixture
def striation_mark_reference() -> Mark:
    return create_synthetic_striation_mark(height=256, width=200, seed=42)


@pytest.fixture
def striation_mark_compared() -> Mark:
    return create_synthetic_striation_mark(height=256, width=220, seed=43)


@pytest.fixture
def striation_mark_reference_aligned() -> Mark:
    return create_synthetic_striation_mark(height=200, width=200, seed=44)


@pytest.fixture
def striation_mark_compared_aligned() -> Mark:
    return create_synthetic_striation_mark(height=200, width=200, seed=45)


@pytest.fixture
def striation_mark_profile_reference() -> Mark:
    return create_synthetic_profile_mark(length=200, seed=46)


@pytest.fixture
def striation_mark_profile_compared() -> Mark:
    return create_synthetic_profile_mark(length=200, seed=47)


@pytest.fixture
def striation_quality_passbands() -> dict[tuple[float, float], float]:
    return {
        (5, 250): 0.85,
        (100, 250): 0.78,
        (50, 100): 0.65,
        (25, 50): 0.45,
        (10, 25): 0.30,
        (5, 10): 0.15,
    }


@pytest.fixture
def striation_metrics(striation_quality_passbands) -> StriationComparisonMetrics:
    return StriationComparisonMetrics(
        score=0.85,
        shift=12.5,
        overlap=80.4,
        sq_ref=0.2395,
        sq_comp=0.7121,
        sq_diff=0.6138,
        sq_ratio=297.3765,
        sign_diff_dsab=220.94,
        data_spacing=1.5625,
        quality_passbands=striation_quality_passbands,
    )


@pytest.fixture
def sample_metadata_reference() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_1",
        "Measurement ID": "striated_mark",
    }


@pytest.fixture
def sample_metadata_compared() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_2",
        "Measurement ID": "striated_mark",
    }


@pytest.fixture
def impression_sample_depth_data() -> FloatArray2D:
    return create_synthetic_impression_data(height=100, width=120, seed=42)


@pytest.fixture
def impression_sample_mark() -> Mark:
    return create_synthetic_impression_mark(height=100, width=120, seed=42)


@pytest.fixture
def impression_sample_cell_correlations() -> np.ndarray:
    """4x5 grid of cell correlation values."""
    return np.random.default_rng(42).random((4, 5))


@pytest.fixture
def impression_sample_metrics(
    impression_sample_cell_correlations: np.ndarray,
) -> ImpressionComparisonMetrics:
    n_cells = impression_sample_cell_correlations.size
    return ImpressionComparisonMetrics(
        area_correlation=0.85,
        cell_correlations=impression_sample_cell_correlations,
        cmc_score=75.0,
        sq_ref=1.5,
        sq_comp=1.6,
        sq_diff=0.4,
        has_area_results=True,
        has_cell_results=True,
        cell_positions_compared=np.full((n_cells, 2), np.nan),
        cell_rotations_compared=np.full(n_cells, np.nan),
        cmc_area_fraction=16.04,
        cutoff_low_pass=5.0,
        cutoff_high_pass=250.0,
        cell_size_um=125.0,
        max_error_cell_position=75.0,
        max_error_cell_angle=6.0,
    )
