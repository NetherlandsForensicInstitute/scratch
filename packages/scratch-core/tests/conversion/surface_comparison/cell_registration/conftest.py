import pytest
from _pytest.fixtures import SubRequest

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import GridCell, ComparisonParams
import numpy as np

from .helpers import (
    make_scan_image,
    make_grid_cell,
    make_surface,
    identity_params,
)


IMAGE_HEIGHT = 980
IMAGE_WIDTH = 720
IMAGE_NAN_RATIO = 0.15
CELL_SIZE = 54
PIXEL_SIZE = 1e-6
CELL_TOP_LEFT = (150, 436)
SCORE_TOLERANCE = 0.05
FILL_FRACTION_THRESHOLD = 0.5


@pytest.fixture(params=[1.0, 1e-6], ids=["unit_scale", "micron_scale"])
def identical_match_inputs(
    request: SubRequest,
) -> tuple[list[GridCell], ScanImage, ComparisonParams]:
    """
    Provide (grid_cells, comparison_image, params) for a self-matching test
    at both unit and µm magnitude scales.
    """
    scale = request.param
    scan_image = make_scan_image(
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        pixel_size=PIXEL_SIZE,
        scale=scale,
        nan_ratio=IMAGE_NAN_RATIO,
    )
    cell_data = scan_image.data[
        CELL_TOP_LEFT[1] : CELL_TOP_LEFT[1] + CELL_SIZE,
        CELL_TOP_LEFT[0] : CELL_TOP_LEFT[0] + CELL_SIZE,
    ]
    nan_fill_value = float(np.nanmean(scan_image.data))  # TODO: Do we want this value?
    grid_cell = make_grid_cell(
        data=cell_data, top_left=CELL_TOP_LEFT, nan_fill_value=nan_fill_value
    )
    params = identity_params(cell_size_px=CELL_SIZE, pixel_size=PIXEL_SIZE)
    return [grid_cell], scan_image, params


@pytest.fixture
def fully_valid_grid_cell() -> GridCell:
    data = make_surface(height=CELL_SIZE, width=CELL_SIZE, scale=1e-6)
    cell = make_grid_cell(data=data, top_left=(10, 10))
    cell.grid_search_params.update(center_x=30, center_y=30, angle=0.0, score=0.95)
    return cell
