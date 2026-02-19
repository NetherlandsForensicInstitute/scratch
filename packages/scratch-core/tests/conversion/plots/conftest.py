import numpy as np
import pytest

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.plots.data_formats import (
    HistogramData,
    ImpressionComparisonMetrics,
    LlrTransformationData,
)
from conversion.profile_correlator import StriationComparisonResults

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
def striation_metrics(striation_quality_passbands) -> StriationComparisonResults:
    sq_ref = 0.2395e-6
    sq_comp = 0.7121e-6
    sq_diff = 0.6138e-6
    return StriationComparisonResults(
        pixel_size=1.5625e-6,
        position_shift=12.5e-6,
        scale_factor=1.0,
        similarity_value=0.85,
        overlap_length=160e-6,
        overlap_ratio=0.804,
        correlation_coefficient=0.85,
        sa_ref=0.19e-6,
        mean_square_ref=sq_ref,
        sa_comp=0.60e-6,
        mean_square_comp=sq_comp,
        sa_diff=0.50e-6,
        mean_square_of_difference=sq_diff,
        ds_roughness_normalized_to_reference=(sq_diff / sq_ref) ** 2,
        ds_roughness_normalized_to_compared=(sq_diff / sq_comp) ** 2,
        ds_roughness_normalized_to_reference_and_compared=sq_diff**2
        / (sq_ref * sq_comp),
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
def impression_sample_mark_reference() -> Mark:
    return create_synthetic_impression_mark(height=100, width=120, seed=42)


@pytest.fixture
def impression_sample_mark_compared() -> Mark:
    return create_synthetic_impression_mark(
        height=100,
        width=120,
        seed=43,
        rotation_deg=15,
    )


@pytest.fixture
def impression_sample_mark_compared_filtered() -> Mark:
    return create_synthetic_impression_mark(
        height=100,
        width=120,
        seed=43,
        rotation_mask_deg=15,
    )


@pytest.fixture
def impression_sample_cell_correlations() -> np.ndarray:
    """4x5 grid of cell correlation values."""
    return np.random.default_rng(42).random((4, 5))


@pytest.fixture
def impression_sample_metrics(
    impression_sample_cell_correlations: np.ndarray,
) -> ImpressionComparisonMetrics:
    n_rows, n_cols = impression_sample_cell_correlations.shape
    n_cells = impression_sample_cell_correlations.size
    # Surface: 100x120 pixels at 1.5e-6 m/px = 150x180 µm
    height_um, width_um = 150.0, 180.0
    cell_w_um = width_um / n_cols
    cell_h_um = height_um / n_rows
    rng = np.random.default_rng(42)
    positions = np.empty((n_cells, 2), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            flat = i * n_cols + j
            positions[flat, 0] = (j + 0.5) * cell_w_um + rng.uniform(-2, 2)
            positions[flat, 1] = (n_rows - 1 - i + 0.5) * cell_h_um + rng.uniform(-2, 2)
    rotations = rng.uniform(-0.05, 0.05, n_cells)
    return ImpressionComparisonMetrics(
        area_correlation=0.85,
        cell_correlations=impression_sample_cell_correlations,
        cmc_score=75.0,
        mean_square_ref=1.5,
        mean_square_comp=1.6,
        mean_square_of_difference=0.4,
        has_area_results=True,
        has_cell_results=True,
        cell_positions_compared=positions,
        cell_rotations_compared=rotations,
        cmc_area_fraction=16.04,
        cutoff_low_pass=5.0,
        cutoff_high_pass=250.0,
        cell_size_um=125.0,
        max_error_cell_position=75.0,
        max_error_cell_angle=6.0,
    )


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
        mean_square_ref=0.1234,
        mean_square_comp=0.1456,
        mean_square_of_difference=0.0567,
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


@pytest.fixture
def ccf_results_metadata() -> dict[str, str]:
    return {
        "Date report": "2023-02-16",
        "User ID": "RUHES (apc_abal)",
        "Mark type": "Aperture shear striation",
        "Score type": "CCF",
        "Score (transform)": "0.97 (1.86)",
        "LogLR (5%, 95%)": "5.19 (5.17, 5.24)",
        "# of KM scores": "1144",
        "# of KNM scores": "296462",
    }


@pytest.fixture
def ccf_histogram_data() -> HistogramData:
    rng = np.random.default_rng(42)
    knm_scores = rng.beta(2, 5, 1000)
    km_scores = rng.beta(8, 2, 100)
    scores = np.concatenate([knm_scores, km_scores])
    labels = np.concatenate([np.zeros(1000), np.ones(100)])
    return HistogramData(
        scores=scores,
        labels=labels,
        bins=None,
        densities=None,
        new_score=None,
    )


@pytest.fixture
def ccf_histogram_data_transformed(ccf_histogram_data: HistogramData) -> HistogramData:
    return HistogramData(
        scores=0.52 + ccf_histogram_data.scores * 0.47,
        labels=ccf_histogram_data.labels,
        bins=None,
        densities=None,
        new_score=None,
    )


@pytest.fixture
def ccf_llr_data(
    ccf_histogram_data_transformed: HistogramData,
) -> LlrTransformationData:
    scores_t = ccf_histogram_data_transformed.scores
    score_grid = np.linspace(scores_t.min(), scores_t.max(), 100)
    llrs = 5 * (score_grid - 0.75) ** 2 - 2
    return LlrTransformationData(
        scores=score_grid,
        llrs=llrs,
        llrs_at5=llrs - 0.5,
        llrs_at95=llrs + 0.5,
        score_llr_point=None,
    )


@pytest.fixture
def cmc_results_metadata(
    impression_overview_metrics: ImpressionComparisonMetrics,
) -> dict[str, str]:
    metrics = impression_overview_metrics
    n_cell_rows, n_cell_cols = metrics.cell_correlations.shape
    n_cells = n_cell_rows * n_cell_cols
    n_cmc = int(np.sum(metrics.cell_correlations >= metrics.cell_similarity_threshold))
    return {
        "Date report": "2023-02-16",
        "User ID": "test_user",
        "Mark type": "Breech face impression",
        "Collection name": "test_collection",
        "KM model": "Beta-binomial",
        "KNM model": "Binomial",
        "Score type": "CMC",
        "Score (transform)": f"{n_cmc} of {n_cells}",
        "LogLR (5%, 95%)": "4.87 (4.87, 4.87)",
        "LR (5%, 95%)": "7.41e+04 (7.41e+04, 7.41e+04)",
        "# of KM scores": "500",
        "# of KNM scores": "5000",
    }


@pytest.fixture
def cmc_histogram_data() -> HistogramData:
    rng = np.random.default_rng(42)
    n_knm, n_km = 5000, 500
    knm_scores = rng.exponential(scale=2.0, size=n_knm)
    km_scores = np.clip(rng.normal(loc=28, scale=5, size=n_km), 0, None)
    scores = np.concatenate([knm_scores, km_scores])
    labels = np.concatenate([np.zeros(n_knm), np.ones(n_km)])
    return HistogramData(
        scores=scores,
        labels=labels,
        bins=20,
        densities=None,
        new_score=None,
    )


@pytest.fixture
def cmc_llr_data() -> LlrTransformationData:
    llr_scores = np.linspace(0, 55, 200)
    llrs = np.piecewise(
        llr_scores,
        [llr_scores < 20, llr_scores >= 20],
        [lambda s: -2 + 0.1 * s, lambda s: -2 + 0.35 * (s - 10)],
    )
    return LlrTransformationData(
        scores=llr_scores,
        llrs=llrs,
        llrs_at5=llrs - 0.3,
        llrs_at95=llrs + 0.3,
        score_llr_point=None,
    )
