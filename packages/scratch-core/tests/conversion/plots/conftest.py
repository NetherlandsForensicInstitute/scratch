import numpy as np
import pytest
from scipy.constants import micro

from container_models.base import FloatArray2D
from conversion.data_formats import Mark, MarkType, MarkMetadata
from conversion.plots.data_formats import (
    HistogramData,
    LlrTransformationData,
)
from conversion.profile_correlator import (
    StriationComparisonResults,
    Profile,
    AlignmentParameters,
)
from conversion.surface_comparison.models import (
    Cell,
    CellMetaData,
    ComparisonParams,
    ComparisonResult,
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
        ds_normalized_combined=sq_diff**2 / (sq_ref * sq_comp),
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
        "reference_raw": make_mark(
            data_ref_lev,
            scale_x=scale_x,
            scale_y=scale_y,
            mark_type=MarkType.EJECTOR_IMPRESSION,
        ),
        "compared_raw": make_mark(
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
def impression_overview_cells() -> list[Cell]:
    """3x4 cell grid with 2 CMC cells and custom positions."""
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
    surface_w_m = cols * scale_x
    surface_h_m = rows * scale_y
    cell_w_m = surface_w_m / n_cell_cols
    cell_h_m = surface_h_m / n_cell_rows

    rng = np.random.default_rng(42)
    global_dx_m, global_dy_m = 8e-6, -6e-6
    global_angle_deg = 2.5

    cells: list[Cell] = []
    for r in range(n_cell_rows):
        for c in range(n_cell_cols):
            score = float(cell_correlations[r, c])
            is_cmc = score >= cell_similarity_threshold

            center_ref = (
                (c + 0.5) * cell_w_m,
                (n_cell_rows - 1 - r + 0.5) * cell_h_m,
            )

            if is_cmc:
                center_comp = (
                    center_ref[0] + global_dx_m + rng.normal(0, 3e-6),
                    center_ref[1] + global_dy_m + rng.normal(0, 3e-6),
                )
                angle_noise = rng.normal(0, 1.5) * (1 - score)
                angle_deg = global_angle_deg + angle_noise
            else:
                center_comp = center_ref
                angle_deg = 0.0
                angle_noise = 0.0

            cells.append(
                Cell(
                    center_reference=center_ref,
                    cell_size=(cell_w_m, cell_h_m),
                    fill_fraction_reference=rng.uniform(0.7, 1.0),
                    best_score=score,
                    angle_deg=angle_deg,
                    center_comparison=center_comp,
                    is_congruent=is_cmc,
                    meta_data=CellMetaData(
                        is_outlier=not is_cmc,
                        residual_angle_deg=angle_noise,
                        position_error=(
                            (
                                center_comp[0] - center_ref[0],
                                center_comp[1] - center_ref[1],
                            )
                            if is_cmc
                            else (0.0, 0.0)
                        ),
                    ),
                )
            )

    return cells


@pytest.fixture
def impression_overview_comparison_params() -> ComparisonParams:
    return ComparisonParams(
        cell_size=(1e-3, 1e-3),
        minimum_fill_fraction=0.5,
        correlation_threshold=0.25,
        angle_deviation_threshold=6.0,
        position_threshold=75e-6,
    )


@pytest.fixture
def impression_overview_cmc_result(
    impression_overview_cells: list[Cell],
) -> ComparisonResult:
    return ComparisonResult(
        cells=impression_overview_cells,
        shared_rotation=2.5,
        shared_translation=(8 * micro, -6 * micro),
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
    impression_overview_cells: list[Cell],
) -> dict[str, str]:
    n_cells = len(impression_overview_cells)
    n_cmc = sum(c.is_congruent for c in impression_overview_cells)
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
