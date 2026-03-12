import numpy as np
import pytest
from scipy.constants import mega, micro

from container_models.base import FloatArray2D
from conversion.data_formats import Mark, MarkType, MarkMetadata
from conversion.plots.data_formats import (
    HistogramData,
    ImpressionComparisonMetrics,
    LlrTransformationData,
)
from conversion.profile_correlator import (
    StriationComparisonResults,
    Profile,
    AlignmentParameters,
)
from .helper_functions import (
    create_synthetic_impression_data,
    create_synthetic_impression_mark,
    create_synthetic_impression_surface_pair,
    create_synthetic_profile,
    create_synthetic_striation_data,
    create_synthetic_striation_mark,
    make_mark,
)


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
def profile_reference() -> Profile:
    return create_synthetic_profile(length=200, seed=46)


@pytest.fixture
def profile_compared() -> Profile:
    return create_synthetic_profile(length=200, seed=47)


@pytest.fixture
def striation_metrics() -> StriationComparisonResults:
    sq_ref = 0.2395 * micro
    sq_comp = 0.7121 * micro
    sq_diff = 0.6138 * micro
    return StriationComparisonResults(
        # Registration
        pixel_size=1.5625 * micro,
        position_shift=12.5 * micro,
        scale_factor=1.0,
        correlation_coefficient=0.85,
        overlap_length=160 * micro,
        overlap_ratio=0.804,
        # Roughness — individual profiles
        sa_ref=0.19 * micro,
        sq_ref=sq_ref,
        sa_comp=0.60 * micro,
        sq_comp=sq_comp,
        # Roughness — difference profile
        sa_diff=0.50 * micro,
        sq_diff=sq_diff,
        # Normalized signature differences
        ds_normalized_ref=(sq_diff / sq_ref) ** 2,
        ds_normalized_comp=(sq_diff / sq_comp) ** 2,
        ds_normalized_combined=sq_diff ** 2 / (sq_ref * sq_comp),
        # Sample-space geometry
        shift_samples=8,
        overlap_samples=102,
        idx_reference_start=8,
        idx_compared_start=0,
        len_reference_equalized=110,
        len_compared_equalized=102,
        # Original profile metadata
        pixel_size_reference=1.5625 * micro,
        pixel_size_compared=1.5625 * micro,
        len_reference_original=110,
        len_compared_original=102,
        # Reproducibility
        alignment_parameters=AlignmentParameters(),
    )


@pytest.fixture
def sample_metadata_reference() -> MarkMetadata:
    return MarkMetadata(
        case_id="firearms",
        firearm_id="firearm_1_-_known_match",
        specimen_id="bullet_1",
        measurement_id="striated_mark",
        mark_id="mark_ref",
    )


@pytest.fixture
def sample_metadata_compared() -> MarkMetadata:
    return MarkMetadata(
        case_id="firearms",
        firearm_id="firearm_1_-_known_match",
        specimen_id="bullet_2",
        measurement_id="striated_mark",
        mark_id="mark_comp",
    )


@pytest.fixture
def impression_sample_depth_data() -> FloatArray2D:
    return create_synthetic_impression_data(height=100, width=120, seed=42)


@pytest.fixture
def impression_sample_mark() -> Mark:
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
def impression_overview_marks() -> dict[str, Mark]:
    """Four impression marks: leveled and filtered for reference and compared."""
    rows, cols = 300, 200
    scale_x = 1.5626 * micro
    scale_y = 1.5675 * micro

    data_ref_lev, data_comp_lev = create_synthetic_impression_surface_pair(
        rows, cols, 0, 10, 11
    )
    data_ref_flt, data_comp_flt = create_synthetic_impression_surface_pair(
        rows, cols, 1, 12, 13
    )

    return {
        "reference_leveled": make_mark(
            data_ref_lev,
            scale_x=scale_x,
            scale_y=scale_y,
            mark_type=MarkType.EJECTOR_IMPRESSION,
        ),
        "compared_leveled": make_mark(
            data_comp_lev,
            scale_x=scale_x,
            scale_y=scale_y,
            mark_type=MarkType.EJECTOR_IMPRESSION,
        ),
        "reference_filtered": make_mark(
            data_ref_flt,
            scale_x=scale_x,
            scale_y=scale_y,
            mark_type=MarkType.EJECTOR_IMPRESSION,
        ),
        "compared_filtered": make_mark(
            data_comp_flt,
            scale_x=scale_x,
            scale_y=scale_y,
            mark_type=MarkType.EJECTOR_IMPRESSION,
        ),
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
    scale_x = 1.5626 * micro
    scale_y = 1.5675 * micro

    n_cell_rows, n_cell_cols = cell_correlations.shape
    n_cells = n_cell_rows * n_cell_cols
    n_cmc = int(np.sum(cell_correlations >= cell_similarity_threshold))
    cmc_score = n_cmc / n_cells * 100

    surface_w_um = cols * scale_x * mega
    surface_h_um = rows * scale_y * mega
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
        cell_correlations=cell_correlations,
        cmc_score=cmc_score,
        cell_positions_compared=positions,
        cell_rotations_compared=rotations,
        cell_similarity_threshold=cell_similarity_threshold,
        cmc_area_fraction=16.04,
        cell_size_um=125.0,
        max_error_cell_position=75.0,
        max_error_cell_angle=6.0,
    )


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
