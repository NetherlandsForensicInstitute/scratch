import numpy as np
import pytest

from container_models.base import FloatArray2D
from conversion.data_formats import Mark
from conversion.plots.data_formats import (
    StriationComparisonMetrics,
    ImpressionComparisonMetrics,
)

from container_models.scan_image import ScanImage
from conversion.data_formats import MarkType

from .helper_functions import (
    create_synthetic_impression_data,
    create_synthetic_impression_mark,
    create_synthetic_impression_surface_pair,
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


# --- Impression overview fixtures ---


@pytest.fixture
def impression_overview_marks() -> dict[str, Mark]:
    """Four impression marks: leveled and filtered for reference and compared."""
    rows, cols = 300, 200
    scale_x = 1.5626e-6
    scale_y = 1.5675e-6

    data_ref_lev, data_comp_lev = create_synthetic_impression_surface_pair(
        rows, cols, 0, 10, 11
    )
    data_ref_flt, data_comp_flt = create_synthetic_impression_surface_pair(
        rows, cols, 1, 12, 13
    )

    def _mark(data: np.ndarray) -> Mark:
        return Mark(
            scan_image=ScanImage(data=data, scale_x=scale_x, scale_y=scale_y),
            mark_type=MarkType.EJECTOR_IMPRESSION,
        )

    return {
        "reference_leveled": _mark(data_ref_lev),
        "compared_leveled": _mark(data_comp_lev),
        "reference_filtered": _mark(data_ref_flt),
        "compared_filtered": _mark(data_comp_flt),
    }


@pytest.fixture
def impression_overview_metrics() -> ImpressionComparisonMetrics:
    """Metrics for a 3x4 cell grid with 2 CMC cells and custom positions."""
    cell_similarity_threshold = 0.25
    cell_correlations = np.array(
        [
            [0.18, 0.09, 0.52, 0.14],
            [0.07, 0.11, 0.06, 0.41],
            [0.13, 0.22, 0.10, 0.05],
        ],
        dtype=np.float64,
    )

    rows, cols = 300, 200
    scale_x = 1.5626e-6
    scale_y = 1.5675e-6

    n_cell_rows, n_cell_cols = cell_correlations.shape
    n_cells = n_cell_rows * n_cell_cols
    n_cmc = int(np.sum(cell_correlations >= cell_similarity_threshold))
    cmc_score = n_cmc / n_cells * 100

    surface_w_um = cols * scale_x * 1e6
    surface_h_um = rows * scale_y * 1e6
    cell_w_um = surface_w_um / n_cell_cols
    cell_h_um = surface_h_um / n_cell_rows

    rng = np.random.default_rng(42)
    positions = np.full((n_cells, 2), np.nan, dtype=np.float64)
    rotations = np.full(n_cells, np.nan, dtype=np.float64)
    global_dx, global_dy = 8.0, -6.0
    global_angle = np.deg2rad(2.5)

    for r in range(n_cell_rows):
        for c in range(n_cell_cols):
            if cell_correlations[r, c] < cell_similarity_threshold:
                continue
            flat = r * n_cell_cols + c
            grid_cx = (c + 0.5) * cell_w_um
            grid_cy = (n_cell_rows - 1 - r + 0.5) * cell_h_um
            positions[flat, 0] = grid_cx + global_dx + rng.normal(0, 3)
            positions[flat, 1] = grid_cy + global_dy + rng.normal(0, 3)
            angle_noise = rng.normal(0, np.deg2rad(1.5)) * (1 - cell_correlations[r, c])
            rotations[flat] = global_angle + angle_noise

    return ImpressionComparisonMetrics(
        area_correlation=0.4123,
        cell_correlations=cell_correlations,
        cmc_score=cmc_score,
        sq_ref=0.1234,
        sq_comp=0.1456,
        sq_diff=0.0567,
        has_area_results=True,
        has_cell_results=True,
        cell_positions_compared=positions,
        cell_rotations_compared=rotations,
        cell_similarity_threshold=cell_similarity_threshold,
        cmc_area_fraction=16.04,
        cutoff_low_pass=5.0,
        cutoff_high_pass=250.0,
        cell_size_um=125.0,
        max_error_cell_position=75.0,
        max_error_cell_angle=6.0,
    )


@pytest.fixture
def impression_overview_metadata_reference() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_known_match",
        "Specimen ID": "kras_1",
        "Measurement ID": "afsloeting_1",
    }


@pytest.fixture
def impression_overview_metadata_compared() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_known_match",
        "Specimen ID": "kras_2_r01",
        "Measurement ID": "afsloeting_2",
    }
