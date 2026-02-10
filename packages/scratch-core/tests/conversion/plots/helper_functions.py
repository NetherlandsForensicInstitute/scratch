import numpy as np

from container_models.base import FloatArray2D, UInt8Array3D
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType


def assert_valid_rgb_image(result: UInt8Array3D) -> None:
    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


def create_synthetic_striation_data(
    height: int = 256,
    width: int = 200,
    seed: int = 42,
) -> FloatArray2D:
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
        data = np.expand_dims(pattern_1d + 0.05 * rng.standard_normal(width), axis=-1)
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


def create_synthetic_striation_mark(
    height: int = 256,
    width: int = 200,
    scale: float = 1.5625e-6,
    seed: int = 42,
) -> Mark:
    """Create a Mark with synthetic striation surface data."""
    return Mark(
        scan_image=ScanImage(
            data=create_synthetic_striation_data(height, width, seed),
            scale_x=scale,
            scale_y=scale,
        ),
        mark_type=MarkType.CHAMBER_STRIATION,
        meta_data={"highpass_cutoff": 5, "lowpass_cutoff": 25},
    )


def create_synthetic_profile_mark(
    length: int = 200,
    scale: float = 1.5625e-6,
    seed: int = 42,
) -> Mark:
    """Create a Mark with synthetic profile data."""
    return Mark(
        scan_image=ScanImage(
            data=create_synthetic_striation_data(height=1, width=length, seed=seed),
            scale_x=scale,
            scale_y=scale,
        ),
        mark_type=MarkType.CHAMBER_STRIATION,
    )


def create_synthetic_impression_data(
    height: int = 100,
    width: int = 120,
    seed: int = 42,
) -> FloatArray2D:
    """
    Create synthetic impression data with horizontal banding.

    :param height: Number of rows.
    :param width: Number of columns.
    :param seed: Random seed for reproducibility.
    :returns: Data in meters with shape (height, width).
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    xn = x / width
    yn = y / height

    surface = (
        2.0 * np.sin(2 * np.pi * yn * 8)
        + 1.2 * np.sin(2 * np.pi * yn * 14 + 1.0)
        + 0.7 * np.cos(2 * np.pi * yn * 22)
    )
    surface *= 1.0 + 0.15 * np.sin(2 * np.pi * xn * 2 + 0.7)
    surface += 0.10 * rng.standard_normal((height, width))

    return (surface * 1e-6).astype(np.float64)


def create_synthetic_impression_mark(
    height: int = 100,
    width: int = 120,
    scale: float = 1.5e-6,
    seed: int = 42,
    mark_type: MarkType = MarkType.FIRING_PIN_IMPRESSION,
) -> Mark:
    """Create a Mark with synthetic impression surface data."""
    return Mark(
        scan_image=ScanImage(
            data=create_synthetic_impression_data(height, width, seed),
            scale_x=scale,
            scale_y=scale,
        ),
        mark_type=mark_type,
    )
