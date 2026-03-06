"""Shared helpers for NIST dataset comparison tests and CMC classification tests."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import Cell, CellMetaData, ComparisonParams

# ---------------------------------------------------------------------------
# NIST dataset helpers
# ---------------------------------------------------------------------------


@dataclass
class MatlabTestCase:
    """A single MATLAB test case with inputs and expected outputs."""

    case_name: str
    reference_map: ScanImage
    comparison_map: ScanImage
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
) -> ScanImage:
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
        kwargs["cell_size"] = (
            params["cell_size_x"] * 1e-6,
            params["cell_size_y"] * 1e-6,
        )

    valid_fields = ComparisonParams.model_fields
    for k, v in params.items():
        # Rename old MATLAB metadata field to the new model field name
        if k == "angle_threshold":
            k = "angle_deviation_threshold"
        if k not in valid_fields:
            continue
        kwargs[k] = float(v) * 1e-6 if k in _UM_FIELDS else float(v)

    if "angle_deviation_threshold" in kwargs and "search_angle_max" not in kwargs:
        kwargs["search_angle_max"] = kwargs["angle_deviation_threshold"]
    if "search_angle_max" in kwargs and "search_angle_min" not in kwargs:
        kwargs["search_angle_min"] = -kwargs["search_angle_max"]

    return ComparisonParams(**kwargs)


# ---------------------------------------------------------------------------
# Classification test helpers
# ---------------------------------------------------------------------------

# Placeholder cell data: a minimal 2x2 height map, unused by the classifier.
_PLACEHOLDER_CELL_DATA = np.array([[0.0, 0.0], [0.1, 0.1]])

# Default CellMetaData used before the classifier has run.
_DEFAULT_META_DATA = CellMetaData(
    is_outlier=False,
    residual_angle_deg=0.0,
    position_error=(0.0, 0.0),
)


def build_cells(inputs: dict) -> list[Cell]:
    """
    Build a list of Cell objects from a test-case input dict.

    Reads the MATLAB-style JSON field names (``mPos1``, ``mPos2``, ``angle1``,
    ``angle2``, ``simVals``). Angles are in degrees (use ``cmc_test_data_degrees.json``).

    The angle stored on each Cell is the delta ``angle2 - angle1``.
    In all current test cases ``angle1`` is zero, so the delta equals ``angle2`` directly.

    NaN/None correlation scores (representing failed registration) are replaced with
    0.0 so they pass the Cell validator; they will still fail the correlation threshold
    check during classification and will not be marked congruent.
    """
    centers_reference = np.array(inputs["mPos1"])
    centers_comparison = np.array(inputs["mPos2"])
    angles_reference = np.atleast_1d(np.array(inputs["angle1"], dtype=float))
    angles_comparison = np.atleast_1d(np.array(inputs["angle2"], dtype=float))

    raw_scores = inputs["simVals"]
    if not isinstance(raw_scores, list):
        raw_scores = [raw_scores]
    correlation_scores = np.atleast_1d(
        np.array(
            [np.nan if s is None else s for s in raw_scores],
            dtype=float,
        )
    )
    # NaN scores cannot pass the Cell field validator (ge=0.0); replace with 0.0
    # so the object can be constructed. They will still be rejected by the
    # classifier since 0.0 < any reasonable correlation_threshold.
    correlation_scores = np.where(np.isnan(correlation_scores), 0.0, correlation_scores)

    if centers_reference.ndim == 1:
        centers_reference = centers_reference.reshape(1, -1)
        centers_comparison = centers_comparison.reshape(1, -1)

    return [
        Cell(
            cell_data=_PLACEHOLDER_CELL_DATA,
            center_reference=(
                float(centers_reference[i, 0]),
                float(centers_reference[i, 1]),
            ),
            center_comparison=(
                float(centers_comparison[i, 0]),
                float(centers_comparison[i, 1]),
            ),
            angle_deg=float(angles_comparison[i] - angles_reference[i]),
            best_score=float(correlation_scores[i]),
            fill_fraction_reference=1.0,
            is_congruent=False,
            meta_data=_DEFAULT_META_DATA.model_copy(),
        )
        for i in range(centers_reference.shape[0])
    ]


def build_test_inputs(
    inputs: dict,
) -> tuple[list[Cell], ComparisonParams, tuple[float, float]]:
    """
    Build the full set of inputs for ``classify_congruent_cells`` from a test-case
    input dict.

    Reads MATLAB-style JSON field names: ``simMin`` → correlation_threshold,
    ``angleMax`` → angle_deviation_threshold (degrees), ``distMax`` → position_threshold,
    ``vCenter`` → rotation_center.

    Returns ``(cells, params, rotation_center)``.
    """
    cells = build_cells(inputs)

    params = ComparisonParams(
        correlation_threshold=float(inputs["simMin"]),
        angle_deviation_threshold=float(inputs["angleMax"]),
        position_threshold=float(inputs["distMax"]),
    )

    rotation_center = (float(inputs["vCenter"][0]), float(inputs["vCenter"][1]))

    return cells, params, rotation_center


def _make_cell(
    angle_deg: float,
    center_reference: tuple[float, float] = (0.0, 0.0),
    center_comparison: tuple[float, float] = (0.0, 0.0),
    best_score: float = 0.8,
    is_outlier: bool = False,
    residual_angle_deg: float = 0.0,
    position_error: tuple[float, float] = (0.0, 0.0),
) -> Cell:
    """Construct a minimal Cell with sensible defaults for unit testing."""
    return Cell(
        center_reference=center_reference,
        cell_data=np.zeros((4, 4)),
        fill_fraction_reference=1.0,
        best_score=best_score,
        angle_deg=angle_deg,
        center_comparison=center_comparison,
        is_congruent=False,
        meta_data=CellMetaData(
            is_outlier=is_outlier,
            residual_angle_deg=residual_angle_deg,
            position_error=position_error,
        ),
    )
