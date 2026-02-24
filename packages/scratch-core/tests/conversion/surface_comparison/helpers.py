"""Shared helpers for NIST dataset comparison tests."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from conversion.surface_comparison.models import ComparisonParams, SurfaceMap

# Pixel dimension conversion: MATLAB stores values in metres, pipeline expects micrometres.
_M_TO_UM = 1e6


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
    height_map = np.load(
        str(case_dir / f"input_{height_data_name}_depth_data.npy")
    ).astype(np.float64)
    pixel_spacing = np.array([scale_x * _M_TO_UM, scale_y * _M_TO_UM], dtype=np.float64)
    rows, cols = height_map.shape
    global_center = np.array(
        [cols * pixel_spacing[0] / 2.0, rows * pixel_spacing[1] / 2.0],
        dtype=np.float64,
    )
    return SurfaceMap(
        height_map=height_map, pixel_spacing=pixel_spacing, global_center=global_center
    )


def _build_comparison_params(params: dict) -> ComparisonParams:
    """Convert a params dict to ComparisonParams."""
    # Assemble cell_size from the separate x/y keys before filtering against
    # dataclass fields — cell_size_x/y are not fields themselves and would
    # otherwise be silently dropped, leaving cell_size at its default [1000, 1000].
    kwargs: dict = {}
    if "cell_size_x" in params and "cell_size_y" in params:
        kwargs["cell_size"] = np.array(
            [params["cell_size_x"], params["cell_size_y"]], dtype=np.float64
        )

    kwargs.update(
        {
            k: float(v)
            for k, v in params.items()
            if k in ComparisonParams.__dataclass_fields__
        }
    )

    if "angle_threshold" in kwargs and "search_angle_max" not in kwargs:
        kwargs["search_angle_max"] = kwargs["angle_threshold"]
    if "search_angle_max" in kwargs and "search_angle_min" not in kwargs:
        kwargs["search_angle_min"] = -kwargs["search_angle_max"]

    return ComparisonParams(**kwargs)
