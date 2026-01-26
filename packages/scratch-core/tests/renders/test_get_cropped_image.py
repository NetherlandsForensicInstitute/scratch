import pytest

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms
from renders.get_cropped_image import get_cropped_image


@pytest.mark.integration
@pytest.mark.parametrize(
    "terms, regression_order",
    [
        (SurfaceTerms.PLANE, 0),
        (SurfaceTerms.PLANE, 1),
        (SurfaceTerms.PLANE, 2),
        (SurfaceTerms.SPHERE, 0),
        (SurfaceTerms.SPHERE, 1),
        (SurfaceTerms.SPHERE, 2),
    ],
)
def test_get_cropped_image(
    scan_image_replica: ScanImage,
    mask_array: MaskArray,
    terms: SurfaceTerms,
    regression_order: int,
):
    result = get_cropped_image(
        scan_image=scan_image_replica,
        mask=mask_array,
        terms=terms,
        regression_order=regression_order,
        cutoff_length=250e-6,
        resampling_factors=(2, 2),
    )
    assert result.shape == (mask_array.shape[0] // 2, mask_array.shape[1] // 2)
