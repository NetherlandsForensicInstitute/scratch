from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.surface_comparison.cell_registration.core import coarse_registration
from conversion.surface_comparison.grid import GridCell, generate_grid
from conversion.surface_comparison.models import ComparisonParams, GridSearchParams
from conversion.surface_comparison.pipeline import compare_surfaces, ProcessedMark
import numpy as np
import pytest
from skimage.transform import rotate

from .cell_registration.helpers import plot_cell_registration_results


# TODO: Remove these dummy tests / fixtures and create real tests


@pytest.fixture(scope="module")
def scan_image() -> ScanImage:
    return ScanImage(
        data=np.zeros(shape=(100, 100), dtype=np.float64), scale_x=1, scale_y=1
    )


@pytest.fixture(scope="module")
def mark(scan_image: ScanImage) -> Mark:
    return Mark(
        scan_image=scan_image,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
    )


@pytest.fixture(scope="module")
def params() -> ComparisonParams:
    return ComparisonParams(cell_size=(20, 20))


@pytest.fixture(scope="module")
def grid_cell() -> GridCell:
    return GridCell(
        top_left=(5, 7),
        cell_data=np.zeros(shape=(20, 20), dtype=np.float64),
        grid_search_params=GridSearchParams(),
    )


def test_compare_surfaces_runs(mark: Mark, params: ComparisonParams):
    processed_mark = ProcessedMark(filtered_mark=mark, leveled_mark=mark)
    results = compare_surfaces(
        refence_mark=processed_mark, comparison_mark=processed_mark, params=params
    )
    assert results


def test_coarse_registration_runs(
    grid_cell: GridCell, scan_image: ScanImage, params: ComparisonParams
):
    cells = coarse_registration(
        grid_cells=[grid_cell],
        comparison_image=scan_image,
        params=params,
    )
    assert cells


def test_generate_grid_runs(scan_image: ScanImage, params: ComparisonParams):
    cells = generate_grid(
        scan_image=scan_image,
        cell_size=params.cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
    )
    assert cells


@pytest.mark.integration
def test_coarse_registration_finds_angle(angle: float = 60, plot: bool = False):
    # Arrange
    scale = 1e-6
    nan_fraction = 0.15
    cell_size = (50, 50)
    image_size = (150, 450)
    image_data = np.random.uniform(size=image_size)
    image_data[np.random.uniform(size=image_data.shape) < nan_fraction] = np.nan
    reference_image = ScanImage(
        data=image_data * scale,
        scale_x=scale,
        scale_y=scale,
    )
    params = ComparisonParams(
        cell_size=(cell_size[0] * scale, cell_size[1] * scale),
        search_angle_min=-90,
        search_angle_max=90,
        search_angle_step=30,
        minimum_fill_fraction=0.5,
    )
    rotated = rotate(
        image=image_data,
        angle=angle,
        order=0,
        resize=True,  # resize=False would clip the content on rectangular images
        cval=np.nan,  # type: ignore
    )
    comparison_image = reference_image.model_copy(update={"data": rotated})

    # Act
    grid_cells = generate_grid(
        scan_image=reference_image,
        cell_size=params.cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
    )
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    if plot:
        plot_cell_registration_results(
            reference_image=reference_image,
            comparison_image=comparison_image,
            cells=cells,
        )

    # Assert
    assert all(c.angle_deg == pytest.approx(angle) for c in cells)
