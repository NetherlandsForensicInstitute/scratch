import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkMetadata, MarkType
from conversion.export.mark import ExportedMarkData
from conversion.profile_correlator import Profile
from lir import FeatureData, InstanceData, LLRData
from lir.lrsystems import LRSystem
from scipy.constants import micro
from scipy.interpolate import interp1d

from processors.schemas import ImpressionLRParameters

RESOURCES = Path(__file__).parent.parent.parent / "packages/scratch-core/tests/resources"


@pytest.fixture
def random_lr_system_path() -> Path:
    """Path to the pre-built random LR system pickle in test resources."""
    return RESOURCES / "random_lr_system.pkl"


class _IdentityLRSystem(LRSystem):
    """Minimal LRSystem that returns the input score as the LLR."""

    def __init__(self) -> None:
        pass

    def apply(self, instances: InstanceData) -> LLRData:
        assert isinstance(instances, FeatureData)
        return LLRData(features=instances.features[:, 0])


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


def _save_mark_and_profile(dir_path: Path, profile: Profile, mark: Mark) -> None:
    """Save mark and profile files to a directory."""
    np.savez(dir_path / "profile.npz", heights=profile.heights, pixel_size=profile.pixel_size)
    np.savez(dir_path / "processed.npz", data=mark.scan_image.data)
    mark_json = ExportedMarkData(
        mark_type=MarkType.EXTRACTOR_STRIATION,
        center=(mark.scan_image.height, mark.scan_image.width),
        scale_x=mark.scan_image.scale_x,
        scale_y=mark.scan_image.scale_y,
    ).model_dump(mode="json")
    mark_json["mark_type"] = "EXTRACTOR_STRIATION"
    for stem in ("processed", "aligned"):
        np.savez(dir_path / f"{stem}.npz", data=mark.scan_image.data)
        with open(dir_path / f"{stem}.json", "w") as f:
            json.dump(mark_json, f)


@pytest.fixture
def mark_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Prepare directories with striation mark and profile files."""
    ref_path = tmp_path / "ref_mark"
    comp_path = tmp_path / "comp_mark"
    ref_path.mkdir()
    comp_path.mkdir()

    profile_ref = _create_dummy_profile()
    profile_comp = _shift_profile(profile_ref, 10.0)
    mark_ref = _striation_mark(profile_ref)
    mark_comp = _striation_mark(profile_comp)

    _save_mark_and_profile(ref_path, profile_ref, mark_ref)
    _save_mark_and_profile(comp_path, profile_comp, mark_comp)

    return ref_path, comp_path


@pytest.fixture
def metadata_reference() -> MarkMetadata:
    return MarkMetadata(case_id="ref-001", firearm_id="fw-1", specimen_id="sp-1", measurement_id="ms-1", mark_id="mk-1")


@pytest.fixture
def metadata_compared() -> MarkMetadata:
    return MarkMetadata(
        case_id="comp-001", firearm_id="fw-2", specimen_id="sp-2", measurement_id="ms-2", mark_id="mk-2"
    )


@pytest.fixture
def base_lr_kwargs(metadata_reference: MarkMetadata, metadata_compared: MarkMetadata) -> dict:
    """Return the common kwargs required by all CalculateLR schemas."""
    return {
        "user_id": "AAAAA",
        "date_report": date(2000, 1, 1),
        "metadata_reference": metadata_reference,
        "metadata_compared": metadata_compared,
    }


@pytest.fixture
def dummy_param() -> ImpressionLRParameters:
    return ImpressionLRParameters(
        area_correlation=0.0,
        cell_correlations=[[0.0]],
        cmc_score=0.0,
        mean_square_ref=0.0,
        mean_square_comp=0.0,
        mean_square_of_difference=0.0,
        has_area_results=False,
        has_cell_results=False,
        cell_positions_compared=[[0.0, 0.0]],
        cell_rotations_compared=[0.0],
        cmc_area_fraction=0.0,
        cutoff_low_pass=0.0,
        cutoff_high_pass=0.0,
        cell_size_um=0.0,
        max_error_cell_position=0.0,
        max_error_cell_angle=0.0,
    )


@pytest.fixture
def impression_kwargs(
    mark_dir_ref: Path,
    mark_dir_comp: Path,
    lr_system_file: Path,
    dummy_param: ImpressionLRParameters,
    base_lr_kwargs: dict,
) -> dict:
    """Return valid baseline kwargs for CalculateLRImpression."""
    return {
        "mark_dir_ref": mark_dir_ref,
        "mark_dir_comp": mark_dir_comp,
        "score": 5,
        "lr_system_path": lr_system_file,
        "n_cells": 10,
        "param": dummy_param,
        **base_lr_kwargs,
    }


@pytest.fixture
def striation_kwargs(
    mark_dir_ref: Path,
    mark_dir_comp: Path,
    lr_system_file: Path,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
) -> dict:
    """Return valid baseline kwargs for CalculateLRStriation."""
    return {
        "mark_dir_ref": mark_dir_ref,
        "mark_dir_ref_aligned": mark_dir_ref,
        "mark_dir_comp": mark_dir_comp,
        "mark_dir_comp_aligned": mark_dir_comp,
        "score": 0.5,
        "lr_system_path": lr_system_file,
        "user_id": "AAAAA",
        "date_report": date(2000, 1, 1),
        "metadata_reference": metadata_reference,
        "metadata_compared": metadata_compared,
    }


@pytest.fixture
def striation_lr_kwargs(
    lr_system_path: Path,
    mark_dirs: tuple[Path, Path],
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
) -> dict:
    """Return valid baseline kwargs for CalculateLRStriation."""
    mark_dir_ref, mark_dir_comp = mark_dirs
    return {
        "mark_dir_ref": mark_dir_ref,
        "mark_dir_ref_aligned": mark_dir_ref,
        "mark_dir_comp": mark_dir_comp,
        "mark_dir_comp_aligned": mark_dir_comp,
        "score": 0.5,
        "lr_system_path": lr_system_path,
        "metadata_reference": metadata_reference,
        "metadata_compared": metadata_compared,
        "user_id": "AAAAA",
        "date_report": date(2000, 1, 1),
    }


@pytest.fixture
def impression_lr_kwargs(
    lr_system_path: Path,
    mark_dirs: tuple[Path, Path],
    dummy_param: ImpressionLRParameters,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
) -> dict:
    """Return valid baseline kwargs for CalculateLRImpression."""
    mark_dir_ref, mark_dir_comp = mark_dirs
    return {
        "mark_dir_ref": mark_dir_ref,
        "mark_dir_comp": mark_dir_comp,
        "score": 3,
        "n_cells": 10,
        "lr_system_path": lr_system_path,
        "param": dummy_param,
        "metadata_reference": metadata_reference,
        "metadata_compared": metadata_compared,
        "user_id": "AAAAA",
        "date_report": date(2000, 1, 1),
    }
