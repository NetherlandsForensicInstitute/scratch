from http import HTTPStatus
from pathlib import Path

import numpy as np
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.export.mark import save_mark
from conversion.export.profile import save_profile
from conversion.profile_correlator import Profile
from conversion.surface_comparison.models import Cell, CellMetaData
from PIL import Image
from pydantic import HttpUrl
from scipy.constants import micro
from scipy.interpolate import interp1d
from starlette.testclient import TestClient


def make_cell(  # noqa: PLR0913
    center_reference: tuple[float, float] = (0.0, 0.0),
    best_score: float = 0.8,
    is_congruent: bool = False,
    angle_deg: float = 0.0,
    center_comparison: tuple[float, float] | None = None,
    cell_size: tuple[float, float] = (50e-6, 50e-6),
    fill_fraction_reference: float = 0.9,
    is_outlier: bool | None = None,
    residual_angle_deg: float = 0.0,
    position_error: tuple[float, float] = (0.0, 0.0),
) -> Cell:
    """Create a Cell instance for testing."""
    if center_comparison is None:
        center_comparison = center_reference
    if is_outlier is None:
        is_outlier = not is_congruent

    return Cell(
        center_reference=center_reference,
        cell_size=cell_size,
        fill_fraction_reference=fill_fraction_reference,
        best_score=best_score,
        angle_deg=angle_deg,
        center_comparison=center_comparison,
        is_congruent=is_congruent,
        meta_data=CellMetaData(
            is_outlier=is_outlier,
            residual_angle_deg=residual_angle_deg,
            position_error=position_error,
        ),
    )


def _save_impression_mark(dir_path: Path, mark: Mark) -> None:
    """Save mark files to a directory in the format load_mark_from_path expects."""
    for stem in ("processed", "leveled", "mark", "aligned"):
        save_mark(mark, dir_path / stem)


def _create_dummy_profile(n_samples: int = 1000) -> Profile:
    """Create a synthetic striation profile for testing."""
    n_striations = 20
    amplitude_um = 0.5
    noise_level = 0.05
    pixel_size_m = 0.5 * micro

    x = np.linspace(0, n_striations * 2 * np.pi, n_samples)
    data = np.sin(x) * amplitude_um * micro
    data += np.sin(2 * x) * amplitude_um * 0.3 * micro
    data += np.sin(0.5 * x) * amplitude_um * 0.5 * micro

    rng = np.random.default_rng()
    data += rng.normal(0, amplitude_um * noise_level * micro, n_samples)

    return Profile(heights=data, pixel_size=pixel_size_m)


def assert_valid_png(path: Path) -> None:
    assert path.exists()
    assert Image.open(path).format == "PNG"


def assert_lr_response_valid(client: TestClient, response) -> None:
    """Assert that an LR endpoint response contains a valid LR and reachable PNG plot."""
    assert response.status_code == HTTPStatus.OK, response.json()
    data = response.json()
    assert isinstance(data["lr"], float)
    assert HttpUrl(data["lr_overview_plot"])
    plot_response = client.get(data["lr_overview_plot"])
    assert plot_response.status_code == HTTPStatus.OK
    assert plot_response.headers["content-type"] == "image/png"


def _shift_profile(profile: Profile, shift_samples: float) -> Profile:
    """Create a shifted version of a profile."""
    data = profile.heights
    n = len(data)
    x_orig = np.arange(n)
    interpolator = interp1d(x_orig, data, kind="linear", fill_value=0, bounds_error=False)
    x_new = x_orig + shift_samples
    new_data = interpolator(x_new)

    rng = np.random.default_rng()
    new_data += rng.normal(0, np.nanstd(data) * 0.01, n)

    return Profile(heights=new_data, pixel_size=profile.pixel_size)


def _striation_mark(profile: Profile, n_cols: int = 50) -> Mark:
    """Build a striation Mark by tiling a profile across columns."""
    data = np.tile(profile.heights[:, np.newaxis], (1, n_cols))
    scan_image = ScanImage(data=data, scale_x=profile.pixel_size, scale_y=profile.pixel_size)
    return Mark(scan_image=scan_image, mark_type=MarkType.BULLET_GEA_STRIATION, center=None)


def _impression_mark(data: np.ndarray) -> Mark:
    """Create an impression mark from 2D surface data."""
    return Mark(
        scan_image=ScanImage(data=data, scale_x=micro, scale_y=micro),
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
    )


def _save_impression_marks(dir_path: Path, mark: Mark) -> None:
    """Save mark and profile files to a directory."""
    for stem in ("processed", "leveled", "mark"):
        save_mark(mark, dir_path / stem)


def _save_striation_mark_and_profile(dir_path: Path, profile: Profile, mark: Mark) -> None:
    """Save mark and profile files to a directory."""
    for stem in ("processed", "aligned", "mark"):
        save_mark(mark, dir_path / stem)
    save_profile(profile, dir_path / "profile")
