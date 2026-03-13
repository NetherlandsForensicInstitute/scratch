from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.surface_comparison.cell_registration import coarse_registration
from conversion.surface_comparison.grid import GridCell, generate_grid
from conversion.surface_comparison.models import ComparisonParams
from conversion.surface_comparison.pipeline import compare_surfaces, ProcessedMark
import numpy as np
import pytest


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
        top_left=(5, 7), cell_data=np.zeros(shape=(20, 20), dtype=np.float64)
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
        fill_value_reference=0.0,
    )
    assert cells


def test_generate_grid_runs(scan_image: ScanImage, params: ComparisonParams):
    cells = generate_grid(
        scan_image=scan_image,
        cell_size=params.cell_size,
        minimum_fill_fraction=params.minimum_fill_fraction,
    )
    assert cells
