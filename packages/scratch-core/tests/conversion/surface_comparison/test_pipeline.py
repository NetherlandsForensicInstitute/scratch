from collections.abc import Callable
from itertools import product

import numpy as np
import pytest
from skimage.transform import rotate

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkImpressionType
from conversion.exceptions import NoValidGridCellsError
from conversion.surface_comparison.cmc_classification_median import (
    classify_congruent_cells_median,
)
from conversion.surface_comparison.cmc_consensus.pipeline import (
    classify_congruent_cells_consensus,
)
from conversion.surface_comparison.grid import GridCell, generate_grid
from conversion.surface_comparison.models import ComparisonParams, GridSearchParams
from conversion.surface_comparison.pipeline import ProcessedMark, compare_surfaces

from .cell_registration.helpers import (
    make_scan_image,
    plot_cell_registration_results,
)

# Pixel size that maps the BREECH_FACE_IMPRESSION cell size (450 µm) onto a 54-pixel cell, so a
# test can pick its cell size in pixels without the mark type having a knob for it.
CELL_54_PIXEL_SIZE = 4.5e-4 / 54
SCORE_TOLERANCE = 0.05


def make_processed_mark(scan_image: ScanImage) -> ProcessedMark:
    """Wrap a scan image as a breech-face :class:`ProcessedMark` ready for `compare_surfaces`."""
    mark = Mark(
        scan_image=scan_image, mark_type=MarkImpressionType.BREECH_FACE_IMPRESSION
    )
    return ProcessedMark(filtered_mark=mark, raw_mark=mark)


# TODO: Remove these dummy tests / fixtures and create real tests


@pytest.fixture(scope="module")
def scan_image() -> ScanImage:
    # Use a scale where the default BREECH_FACE_IMPRESSION cell size (450 µm) maps to ~45 pixels
    # 4.5e-4 m / 1e-5 m/pixel = 45 pixels
    return ScanImage(
        data=np.zeros(shape=(100, 100), dtype=np.float64), scale_x=1e-5, scale_y=1e-5
    )


@pytest.fixture(scope="module")
def mark(scan_image: ScanImage) -> Mark:
    return Mark(
        scan_image=scan_image,
        mark_type=MarkImpressionType.BREECH_FACE_IMPRESSION,
    )


@pytest.fixture(scope="module")
def params() -> ComparisonParams:
    return ComparisonParams()


@pytest.fixture(scope="module")
def grid_cell() -> GridCell:
    return GridCell(
        top_left=(5, 7),
        cell_data=np.zeros(shape=(20, 20), dtype=np.float64),
        grid_search_params=GridSearchParams(),
    )


def test_compare_surfaces_runs(mark: Mark, params: ComparisonParams):
    processed_mark = ProcessedMark(filtered_mark=mark, raw_mark=mark)
    results = compare_surfaces(
        reference_mark=processed_mark, comparison_mark=processed_mark, params=params
    )
    assert results


def test_compare_surfaces_runs_the_coarse_stage_when_images_exceed_the_coarse_target():
    # Larger than the default coarse_target_size=256, so cap_factor > 1.0 and compare_surfaces takes the
    # coarse-stage branch (build_coarse_stage, search_candidates, run_fine_stage) rather than the
    # single-exhaustive-pass shortcut that the smaller `mark` fixture exercises above.
    scan_image = ScanImage(
        data=np.zeros(shape=(300, 300), dtype=np.float64), scale_x=1e-5, scale_y=1e-5
    )

    results = compare_surfaces(
        reference_mark=make_processed_mark(scan_image),
        comparison_mark=make_processed_mark(scan_image),
        params=ComparisonParams(),
    )

    assert results


def test_compare_surfaces_raises_on_no_grid_cells():
    # An all-NaN image leaves no cell meeting minimum_fill_fraction, so generate_grid raises an error
    scan_image = ScanImage(
        data=np.full((100, 100), np.nan, dtype=np.float64), scale_x=1e-5, scale_y=1e-5
    )

    with pytest.raises(NoValidGridCellsError):
        _ = compare_surfaces(
            reference_mark=make_processed_mark(scan_image),
            comparison_mark=make_processed_mark(scan_image),
            params=ComparisonParams(),
        )


class TestSelfMatch:
    """Comparing a mark against itself must recover every cell exactly where it came from."""

    @pytest.fixture(params=[1.0, 1e-6], ids=["unit_scale", "micron_scale"])
    def result(self, request: pytest.FixtureRequest):
        # A 980x720 image at this pixel size gives 54-pixel cells and, against the default
        # coarse_target_size of 256, a cap factor above 1.0 - so this runs the real coarse-to-fine path.
        scan_image = make_scan_image(
            height=980,
            width=720,
            pixel_size=CELL_54_PIXEL_SIZE,
            scale=request.param,
            nan_ratio=0.15,
        )
        params = ComparisonParams(
            minimum_fill_fraction=0.5,
            correlation_threshold=0.5,
            search_angle_min=-60.0,
            search_angle_max=60.0,
            search_angle_step=60.0,
        )
        return compare_surfaces(
            reference_mark=make_processed_mark(scan_image),
            comparison_mark=make_processed_mark(scan_image),
            params=params,
        )

    def test_returns_cells(self, result):
        assert result.cells

    def test_score_is_near_one(self, result):
        assert all(cell.best_score >= 1.0 - SCORE_TOLERANCE for cell in result.cells)

    def test_angle_is_zero(self, result):
        assert all(cell.angle_deg == pytest.approx(0.0) for cell in result.cells)

    def test_center_is_unchanged(self, result):
        assert all(
            cell.center_comparison == cell.center_reference for cell in result.cells
        )


@pytest.mark.parametrize(
    "coarse_target_size", [1000, 800], ids=["coarse_target_1000", "coarse_target_800"]
)
@pytest.mark.parametrize("angle", [0, 60, -40])
def test_compare_surfaces_recovers_a_large_rotation(
    angle: float, coarse_target_size: int
):
    # Arrange: a sweep far wider than production uses, since marks are occasionally presented
    # at a wholly different orientation.
    reference_image = make_scan_image(
        height=400, width=360, pixel_size=CELL_54_PIXEL_SIZE, nan_ratio=0.15
    )
    # order=0 (nearest) does no arithmetic across the NaN boundary, so the 15% scattered NaN
    # stays at 15%; bilinear would smear each hole into its neighbors and inflate it to ~48%,
    # starving half the cells of data. It is also how the pipeline itself rotates - see the
    # INTER_NEAREST in cell_registration.geometry._warp. resize=True keeps the canvas from
    # clipping any cell feature away.
    rotated = rotate(
        reference_image.data,
        angle=angle,
        order=0,
        resize=True,
        cval=np.nan,  # type: ignore[arg-type]
    )
    comparison_image = reference_image.model_copy(update={"data": rotated})
    params = ComparisonParams(
        search_angle_min=-80,
        search_angle_max=80,
        search_angle_step=20,
        minimum_fill_fraction=0.5,
        coarse_target_size=coarse_target_size,
    )

    # Act
    result = compare_surfaces(
        reference_mark=make_processed_mark(reference_image),
        comparison_mark=make_processed_mark(comparison_image),
        params=params,
    )

    # Assert
    assert result.cells
    assert all(cell.angle_deg == pytest.approx(angle) for cell in result.cells)


def test_template_nan_fill_strategy_local_mean():
    """Verify local_mean strategy fills NaNs with each cell's own mean."""
    # Arrange
    scale = 1e-6
    rng = np.random.default_rng(seed=42)
    # Create reference image with NaNs
    data = rng.uniform(0.1, 0.9, size=(100, 100)) * scale
    data[10:30, 10:30] = np.nan  # block of NaNs in top-left region
    data[60:80, 60:80] = np.nan  # block of NaNs in bottom-right region
    reference_image = ScanImage(data=data, scale_x=scale, scale_y=scale)
    cell_size = (20 * scale, 20 * scale)
    params = ComparisonParams(
        template_nan_fill_strategy="local_mean",
        minimum_fill_fraction=0.5,
    )

    # Act
    # With local_mean, nan_fill_value should be None
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
        nan_fill_value=None,
    )

    # Assert
    # Find cells with NaNs and verify they use their own local mean
    cells_with_nans = [c for c in grid_cells if np.any(np.isnan(c.cell_data))]
    assert len(cells_with_nans) > 0, "Expected some cells to contain NaNs"

    for cell in cells_with_nans:
        local_mean = float(np.nanmean(cell.cell_data))
        filled = cell.cell_data_filled
        # All NaN positions should be filled with the cell's own local mean
        nan_mask = np.isnan(cell.cell_data)
        assert np.all(filled[nan_mask] == local_mean), (
            f"local_mean strategy: NaN positions not filled with cell mean. "
            f"Expected {local_mean}, got unique values: {np.unique(filled[nan_mask])}"
        )


def test_template_nan_fill_strategy_global_mean():
    """Verify global_mean strategy fills NaNs with the reference image's global mean."""
    # Arrange
    scale = 1e-6
    rng = np.random.default_rng(seed=42)
    # Create reference image with NaNs
    data = rng.uniform(0.1, 0.9, size=(100, 100)) * scale
    data[10:30, 10:30] = np.nan  # block of NaNs in top-left region
    data[60:80, 60:80] = np.nan  # block of NaNs in bottom-right region
    reference_image = ScanImage(data=data, scale_x=scale, scale_y=scale)
    cell_size = (20 * scale, 20 * scale)
    params = ComparisonParams(
        template_nan_fill_strategy="global_mean",
        minimum_fill_fraction=0.5,
    )
    # With global_mean, nan_fill_value should be the global mean of the reference image
    global_mean = float(np.nanmean(reference_image.data))

    # Act
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
        nan_fill_value=global_mean,
    )

    # Assert
    # Find cells with NaNs and verify they all use the same global mean
    cells_with_nans = [c for c in grid_cells if np.any(np.isnan(c.cell_data))]
    assert len(cells_with_nans) > 0, "Expected some cells to contain NaNs"

    for cell in cells_with_nans:
        filled = cell.cell_data_filled
        nan_mask = np.isnan(cell.cell_data)
        assert np.all(filled[nan_mask] == global_mean), (
            f"global_mean strategy: NaN positions not filled with global mean. "
            f"Expected {global_mean}, got unique values: {np.unique(filled[nan_mask])}"
        )


def test_generate_grid_runs(scan_image: ScanImage, params: ComparisonParams):
    # cell_size in meters; with scale_x=1e-5, (2e-4, 2e-4) m = 20 pixels
    cells = generate_grid(
        scan_image=scan_image,
        cell_size=(2e-4, 2e-4),
        minimum_fill_fraction=params.minimum_fill_fraction,
    )
    assert cells


@pytest.mark.integration
@pytest.mark.parametrize(
    ("classification_function", "angle"),
    list(
        product(
            [classify_congruent_cells_consensus, classify_congruent_cells_median],
            [120, 60, -90, -150],
        )
    ),
    ids=lambda x: x.__name__ if callable(x) else str(x),
)
def test_coarse_registration_finds_angle(
    angle: float,
    classification_function: Callable,
    plot: bool = False,
):
    # Arrange
    height_scale = 1e-6
    # 450 µm / 50 px, so the mark type's cell size lands on the 50-pixel cells this test wants.
    pixel_size = 4.5e-4 / 50
    nan_fraction = 0.15
    image_size = (150, 450)
    rng = np.random.default_rng(seed=1234)
    image_data = rng.uniform(size=image_size)
    # Add noisy NaN values to simulate measurement artifacts
    image_data[rng.uniform(size=image_data.shape) < nan_fraction] = np.nan
    # Remove a rectangular part of the image to simulate masking
    image_data[52:112, 157:278] = np.nan
    reference_image = ScanImage(
        data=image_data * height_scale,
        scale_x=pixel_size,
        scale_y=pixel_size,
    )
    params = ComparisonParams(
        search_angle_step=30,
        minimum_fill_fraction=0.5,
    )
    rotated = rotate(
        image=image_data,
        angle=angle,
        order=0,
        resize=True,
        cval=np.nan,  # type: ignore
    )
    comparison_image = reference_image.model_copy(update={"data": rotated})

    # Act
    cells = compare_surfaces(
        reference_mark=make_processed_mark(reference_image),
        comparison_mark=make_processed_mark(comparison_image),
        params=params,
    ).cells

    classification = classification_function(
        cells=cells, params=params, reference_center=reference_image.center_meters
    )

    if plot:
        plot_cell_registration_results(
            reference_image=reference_image,
            comparison_image=comparison_image,
            cells=cells,
        )

    # Assert
    assert all(cell.is_congruent for cell in classification.cells)
    assert all(c.angle_deg == pytest.approx(angle) for c in cells)
