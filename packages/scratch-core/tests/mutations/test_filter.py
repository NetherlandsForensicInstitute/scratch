from pathlib import Path


from computations.spatial import get_bounding_box
from container_models.scan_image import ScanImage
from mutations import GaussianRegressionFilter
from utils.constants import RegressionOrder
import numpy as np
from scipy.constants import micro


def test_gaussian_filter_mutation(baseline_images_dir: Path):
    # Arrange
    filter = GaussianRegressionFilter(
        cutoff_length=250 * micro,
        regression_order=RegressionOrder.LOCAL_QUADRATIC,
        is_high_pass=True,
    )
    rng = np.random.default_rng(4321)
    data = np.full(shape=(500, 1000), fill_value=np.nan, dtype=np.float64)
    valid_region = slice(200, 400), slice(500, 800)
    data[valid_region] = rng.random(size=(200, 300), dtype=np.float64) * micro
    scan_image_masked = ScanImage(data=data, scale_x=micro, scale_y=micro)
    verified_file = baseline_images_dir / "gaussian_filtered.npz"
    verified_array = np.load(verified_file, allow_pickle=True)["array"]

    # Act
    mutated = filter(scan_image_masked)
    bounding_box = get_bounding_box(mutated.valid_mask, margin=0)

    # Assert
    assert np.allclose(
        mutated.data, verified_array, equal_nan=True, rtol=0.0, atol=1e-20
    ), "Filtered image does not match verified image"
    assert np.array_equal(bounding_box, valid_region), (
        "Valid region should be unaffected"
    )
