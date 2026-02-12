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
    rotation_deg: float = 0.0,
    rotation_mask_deg: float = 0.0,
) -> FloatArray2D:
    """
    Create synthetic impression data with banding.

    :param height: Number of rows.
    :param width: Number of columns.
    :param seed: Random seed for reproducibility.
    :param rotation_deg: Rotate the band pattern. Simulates a surface that
        has not yet been aligned to the reference.
    :param rotation_mask_deg: Apply a rotated rectangular NaN mask. Simulates
        a surface that was rotated to align with the reference, leaving NaN
        corners where data is missing.
    :returns: Data in meters with shape (height, width).
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    xn = x / width
    yn = y / height

    angle = np.deg2rad(rotation_deg)
    yn_rot = yn * np.cos(angle) + xn * np.sin(angle)

    surface = (
        2.0 * np.sin(2 * np.pi * yn_rot * 8)
        + 1.2 * np.sin(2 * np.pi * yn_rot * 14 + 1.0)
        + 0.7 * np.cos(2 * np.pi * yn_rot * 22)
    )
    surface *= 1.0 + 0.15 * np.sin(2 * np.pi * xn * 2 + 0.7)
    surface += 0.10 * rng.standard_normal((height, width))

    result = (surface * 1e-6).astype(np.float64)

    if rotation_mask_deg != 0.0:
        mask_angle = np.deg2rad(rotation_mask_deg)
        cx, cy = width / 2, height / 2
        dx = x - cx
        dy = y - cy
        cos_a, sin_a = np.cos(mask_angle), np.sin(mask_angle)
        rx = cos_a * dx + sin_a * dy
        ry = -sin_a * dx + cos_a * dy
        result[(np.abs(rx) > cx) | (np.abs(ry) > cy)] = np.nan

    return result


def create_synthetic_impression_surface_pair(
    rows: int,
    cols: int,
    base_seed: int,
    noise_seed_a: int,
    noise_seed_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a pair of correlated impression surfaces with independent noise.

    Both surfaces share the same base pattern (sinusoidal + Gaussian blobs)
    but receive different noise realisations, simulating two measurements
    of the same surface.

    :param rows: Number of rows.
    :param cols: Number of columns.
    :param base_seed: Seed for the base pattern (Gaussian blob placement).
    :param noise_seed_a: Seed for the first surface's noise.
    :param noise_seed_b: Seed for the second surface's noise.
    :returns: Tuple of two float64 arrays in meters, shape (rows, cols).
    """
    rng = np.random.default_rng(base_seed)
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / cols
    yn = y / rows

    surface = (
        2.0 * np.sin(2 * np.pi * yn * 8)
        + 1.2 * np.sin(2 * np.pi * yn * 14 + 1.0)
        + 0.7 * np.cos(2 * np.pi * yn * 22)
        + 0.4 * np.sin(2 * np.pi * yn * 35 + 0.3)
    )
    surface *= 1.0 + 0.15 * np.sin(2 * np.pi * xn * 2 + 0.7)
    surface += 1.0 * (yn - 0.5) + 0.5 * ((xn - 0.5) ** 2)

    for _ in range(6):
        cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        amp = rng.uniform(-1.5, 1.5)
        sigma = rng.uniform(0.03, 0.08)
        surface += amp * np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / (2 * sigma**2))

    noise_a = np.random.default_rng(noise_seed_a).normal(0, 0.25, (rows, cols))
    noise_b = np.random.default_rng(noise_seed_b).normal(0, 0.25, (rows, cols))
    return (
        ((surface + noise_a) * 1e-6).astype(np.float64),
        ((surface + noise_b) * 1e-6).astype(np.float64),
    )


def create_synthetic_impression_mark(
    height: int = 100,
    width: int = 120,
    scale: float = 1.5e-6,
    seed: int = 42,
    mark_type: MarkType = MarkType.FIRING_PIN_IMPRESSION,
    rotation_deg: float = 0.0,
    rotation_mask_deg: float = 0.0,
) -> Mark:
    """Create a Mark with synthetic impression surface data."""
    return Mark(
        scan_image=ScanImage(
            data=create_synthetic_impression_data(
                height, width, seed, rotation_deg, rotation_mask_deg
            ),
            scale_x=scale,
            scale_y=scale,
        ),
        mark_type=mark_type,
    )
