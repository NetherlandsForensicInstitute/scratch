from itertools import product
from typing import Callable
from conversion.surface_comparison.grid import generate_grid
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkImpressionType
from conversion.surface_comparison.cell_registration.match_cells import match_cells
from conversion.surface_comparison.cmc_consensus.pipeline import (
    classify_congruent_cells_consensus,
)
from conversion.surface_comparison.cmc_classification_median import (
    classify_congruent_cells_median,
)
from conversion.surface_comparison.grid import GridCell
from conversion.surface_comparison.models import (
    ComparisonParams,
    GridSearchParams,
)
from conversion.surface_comparison.pipeline import (
    DOWNSAMPLE_INTERPOLATION,
    UPSAMPLE_INTERPOLATION,
    compare_surfaces,
    ProcessedMark,
    select_interpolation,
)
import numpy as np
import pytest
from skimage.transform import rotate

from .cell_registration.helpers import plot_cell_registration_results


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


@pytest.mark.parametrize(
    ("factors", "expected"),
    [
        ((2.0, 2.0), DOWNSAMPLE_INTERPOLATION),
        # An axis that is left alone must not force upsample handling.
        ((1.0, 3.0), DOWNSAMPLE_INTERPOLATION),
        ((0.5, 0.5), UPSAMPLE_INTERPOLATION),
        # cv2 takes one flag for both axes, so a growing axis wins.
        ((0.5, 2.0), UPSAMPLE_INTERPOLATION),
    ],
)
def test_selects_interpolation_from_the_resampling_direction(
    factors: tuple[float, float], expected: str
):
    assert select_interpolation(factors) == expected


def test_compare_surfaces_runs(mark: Mark, params: ComparisonParams):
    processed_mark = ProcessedMark(filtered_mark=mark, raw_mark=mark)
    results = compare_surfaces(
        reference_mark=processed_mark, comparison_mark=processed_mark, params=params
    )
    assert results


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
    scale = 1e-6
    nan_fraction = 0.15
    cell_size = (50, 50)
    image_size = (150, 450)
    rng = np.random.default_rng(seed=1234)
    image_data = rng.uniform(size=image_size)
    # Add noisy NaN values to simulate measurement artifacts
    image_data[rng.uniform(size=image_data.shape) < nan_fraction] = np.nan
    # Remove a rectangular part of the image to simulate masking
    image_data[52:112, 157:278] = np.nan
    reference_image = ScanImage(
        data=image_data * scale,
        scale_x=scale,
        scale_y=scale,
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
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=(cell_size[0] * scale, cell_size[1] * scale),
        minimum_fill_fraction=params.minimum_fill_fraction,
    )
    cells = match_cells(
        grid_cells=grid_cells,
        reference_image=reference_image,
        comparison_image=comparison_image,
        params=params,
    )

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
