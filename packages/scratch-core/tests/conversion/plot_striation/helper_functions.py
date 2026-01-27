import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType, CropType
from conversion.plots.data_formats import CorrelationMetrics


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
        Y = y[:, np.newaxis]

        data = (
            0.50 * np.sin(2.5 * Y)
            + 0.30 * np.sin(10 * Y)
            + 0.10 * np.sin(33 * Y)
            + 0.20 * np.sin(1.3 * x)
            + 0.10 * rng.standard_normal((height, width))
        )

    return data * 1e-6


def create_mock_mark(
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


def create_mock_profile_mark(
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


@pytest.fixture
def profile_ref() -> np.ndarray:
    return create_synthetic_striation_data(height=1, width=200, seed=42)


@pytest.fixture
def profile_comp() -> np.ndarray:
    return create_synthetic_striation_data(height=1, width=200, seed=43)


@pytest.fixture
def surface_ref() -> np.ndarray:
    return create_synthetic_striation_data(height=256, width=200, seed=42)


@pytest.fixture
def surface_comp() -> np.ndarray:
    return create_synthetic_striation_data(height=256, width=220, seed=43)


@pytest.fixture
def mark_ref() -> Mark:
    return create_mock_mark(height=256, width=200, seed=42)


@pytest.fixture
def mark_comp() -> Mark:
    return create_mock_mark(height=256, width=220, seed=43)


@pytest.fixture
def mark_ref_aligned() -> Mark:
    return create_mock_mark(height=200, width=200, seed=44)


@pytest.fixture
def mark_comp_aligned() -> Mark:
    return create_mock_mark(height=200, width=200, seed=45)


@pytest.fixture
def profile_mark_ref() -> Mark:
    return create_mock_profile_mark(length=200, seed=46)


@pytest.fixture
def profile_mark_comp() -> Mark:
    return create_mock_profile_mark(length=200, seed=47)


@pytest.fixture
def quality_passbands() -> dict[tuple[float, float], float]:
    return {
        (5, 250): 0.85,
        (100, 250): 0.78,
        (50, 100): 0.65,
        (25, 50): 0.45,
        (10, 25): 0.30,
        (5, 10): 0.15,
    }


@pytest.fixture
def metrics(quality_passbands) -> CorrelationMetrics:
    return CorrelationMetrics(
        score=0.85,
        shift=12.5,
        overlap=80.4,
        sq_a=0.2395,
        sq_b=0.7121,
        sq_b_minus_a=0.6138,
        sq_ratio=297.3765,
        sign_diff_dsab=220.94,
        data_spacing=1.5625,
        quality_passbands=quality_passbands,
    )


@pytest.fixture
def metadata_ref() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_1",
        "Measurement ID": "striated_mark",
    }


@pytest.fixture
def metadata_comp() -> dict[str, str]:
    return {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_2",
        "Measurement ID": "striated_mark",
    }


def assert_valid_rgb_image(result: np.ndarray) -> None:
    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
