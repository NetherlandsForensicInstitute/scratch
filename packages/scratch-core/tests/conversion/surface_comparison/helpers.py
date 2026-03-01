"""Shared helpers for NIST dataset comparison tests."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams, SurfaceMap


@dataclass
class MatlabTestCase:
    """A single MATLAB test case with inputs and expected outputs."""

    case_name: str
    reference_map: SurfaceMap
    comparison_map: SurfaceMap
    params: ComparisonParams
    expected_results: dict


def load_test_case(case_dir: Path) -> MatlabTestCase:
    """Load a test case from a converted directory."""
    with open(case_dir / "metadata.json") as f:
        meta = json.load(f)

    reference_map = _load_surface_map(
        case_dir,
        height_data_name="DataConProRef",
        scale_x=float(meta.get("reference_scale_x") or 3.5e-6),
        scale_y=float(meta.get("reference_scale_y") or 3.5e-6),
    )
    comparison_map = _load_surface_map(
        case_dir,
        height_data_name="DataConProCom",
        scale_x=float(meta.get("comparison_scale_x") or 3.5e-6),
        scale_y=float(meta.get("comparison_scale_y") or 3.5e-6),
    )

    return MatlabTestCase(
        case_name=meta["case_name"],
        reference_map=reference_map,
        comparison_map=comparison_map,
        params=_build_comparison_params(meta.get("params", {})),
        expected_results=meta.get("expected_results", {}),
    )


def _load_surface_map(
    case_dir: Path, height_data_name: str, scale_x: float, scale_y: float
) -> SurfaceMap:
    """Load a surface map from a .npy depth file and pixel scale dimensions."""
    data = np.load(str(case_dir / f"input_{height_data_name}_depth_data.npy")).astype(
        np.float64
    )
    return ScanImage(data=data, scale_x=scale_x, scale_y=scale_y)


# Fields in ComparisonParams whose MATLAB metadata values are in µm and must
# be converted to meters before construction.
_UM_FIELDS = {"position_threshold"}


def _build_comparison_params(params: dict) -> ComparisonParams:
    """Convert a params dict (MATLAB metadata, lengths in µm) to ComparisonParams (meters)."""
    kwargs: dict = {}

    # cell_size_x/y arrive in µm from MATLAB metadata; convert to meters.
    if "cell_size_x" in params and "cell_size_y" in params:
        kwargs["cell_size"] = np.array(
            [params["cell_size_x"] * 1e-6, params["cell_size_y"] * 1e-6],
            dtype=np.float64,
        )

    valid_fields = ComparisonParams.model_fields
    for k, v in params.items():
        if k not in valid_fields:
            continue
        kwargs[k] = float(v) * 1e-6 if k in _UM_FIELDS else float(v)

    return ComparisonParams(**kwargs)
