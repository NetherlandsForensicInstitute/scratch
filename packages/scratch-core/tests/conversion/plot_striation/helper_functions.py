import numpy as np

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType, CropType


def create_synthetic_striation_data(
    height: int = 256,
    width: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Create synthetic striation data with horizontal grooves.

    :param height: Number of rows (1 for profile data).
    :param width: Number of columns.
    :param seed: Random seed for reproducibility.
    :returns: Data in meters with shape (height, width) or (width,) if height=1.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, width)

    # Base pattern: sum of sine waves at different frequencies
    pattern_1d = 0.50 * np.sin(2.5 * x) + 0.30 * np.sin(10 * x) + 0.10 * np.sin(33 * x)

    if height == 1:
        data = pattern_1d + 0.05 * rng.standard_normal(width)
    else:
        y = np.linspace(0, 2 * np.pi, height)
        y = y[:, np.newaxis]

        data = (
            0.50 * np.sin(2.5 * y)
            + 0.30 * np.sin(10 * y)
            + 0.10 * np.sin(33 * y)
            + 0.20 * np.sin(1.3 * x)
            + 0.10 * rng.standard_normal((height, width))
        )

    return data * 1e-6


def create_synthetic_mark(
    height: int = 256,
    width: int = 200,
    scale: float = 1.5625e-6,
    seed: int = 42,
) -> Mark:
    """Create a MockMark with synthetic surface data."""
    return Mark(
        scan_image=ScanImage(
            data=create_synthetic_striation_data(height, width, seed),
            scale_x=scale,
            scale_y=scale,
        ),
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
        meta_data={"highpass_cutoff": 5, "lowpass_cutoff": 25},
    )


def create_synthetic_profile_mark(
    length: int = 200,
    scale: float = 1.5625e-6,
    seed: int = 42,
) -> Mark:
    """Create a MockMark with synthetic profile data."""
    return Mark(
        scan_image=ScanImage(
            data=create_synthetic_striation_data(height=1, width=length, seed=seed),
            scale_x=scale,
            scale_y=scale,
        ),
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
    )


def assert_valid_rgb_image(result: np.ndarray) -> None:
    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
