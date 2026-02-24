"""Tests for CompareDatasetsNIST: compare Python implementation against MATLAB reference.

Usage:
    pytest test_compare_datasets_nist.py -v

Requires converted test cases in TEST_DATA_DIR (see convert_comparedatasetsnist_tests.py).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from conversion.surface_comparison.pipeline import run_comparison_pipeline
from conversion.surface_comparison.models import (
    SurfaceMap,
    ComparisonParams,
    ComparisonResult,
)

# ---- Configure test data directory ----
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "resources" / "cmc"

# ---- Tolerances ----
RTOL_SCALAR = 1e-6
ATOL_SCALAR = 1e-10


@dataclass
class SurfaceContainer:
    """Minimal surface data container for testing: depth map and pixel dimensions."""

    depth_data: np.ndarray
    xdim: float
    ydim: float


@dataclass
class MatlabTestCase:
    """A single MATLAB test case with inputs and expected outputs."""

    case_name: str
    data_con_pro_ref: SurfaceContainer
    data_con_pro_com: SurfaceContainer
    param: dict
    expected_res: dict

    @classmethod
    def from_directory(cls, case_dir: Path) -> "MatlabTestCase":
        """Load a test case from a converted directory."""
        meta_path = case_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {case_dir}")

        with open(meta_path) as f:
            meta = json.load(f)

        if "error" in meta:
            raise pytest.skip(
                f"MATLAB error: {meta['error'].get('message', 'unknown')}"
            )

        # Load the two processed (filtered) surface containers used by the pipeline.
        # The levelled containers (DataConLev*) are not used by the Python pipeline.
        containers = {}
        for cname in ["DataConProRef", "DataConProCom"]:
            depth_path = case_dir / f"input_{cname}_depth_data.npy"
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing {depth_path}")
            xdim = meta.get(f"{cname}_xdim", 3.5e-6)
            ydim = meta.get(f"{cname}_ydim", 3.5e-6)
            containers[cname] = SurfaceContainer(
                depth_data=np.load(str(depth_path)),
                xdim=float(xdim) if xdim is not None else 3.5e-6,
                ydim=float(ydim) if ydim is not None else 3.5e-6,
            )

        return cls(
            case_name=meta["case_name"],
            data_con_pro_ref=containers["DataConProRef"],
            data_con_pro_com=containers["DataConProCom"],
            param=meta.get("param", {}),
            expected_res=_restore_nans(meta.get("output_Res", {})),
        )


def _restore_nans(d: dict) -> dict:
    """Convert None values back to NaN for numeric fields."""
    return {k: float("nan") if v is None else v for k, v in d.items()}


def discover_test_cases() -> list[str]:
    """Find all available test case directories."""
    if not TEST_DATA_DIR.exists():
        return []
    return sorted(
        d.name
        for d in TEST_DATA_DIR.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )


def load_test_case(case_name: str) -> MatlabTestCase:
    """Load a single test case by name."""
    return MatlabTestCase.from_directory(TEST_DATA_DIR / case_name)


# ---- Adapter: MATLAB DataContainers → Python SurfaceMaps + ComparisonParams ----

# MATLAB stores pixel dimensions in meters; the pipeline works in micrometers.
_M_TO_UM = 1e6


def _build_surface_map(container: SurfaceContainer) -> SurfaceMap:
    """
    Construct a SurfaceMap from a processed MATLAB surface container.

    MATLAB pixel dimensions (xdim/ydim) are given in **metres**; they are
    converted to micrometres here so all downstream arithmetic is consistent.

    :param container: Processed (filtered) surface container.
    :returns: A fully populated SurfaceMap.
    """
    pixel_spacing = np.array(
        [container.xdim * _M_TO_UM, container.ydim * _M_TO_UM], dtype=np.float64
    )

    height_map = container.depth_data.astype(np.float64)

    rows, cols = height_map.shape
    global_center = np.array(
        [cols * pixel_spacing[0] / 2.0, rows * pixel_spacing[1] / 2.0],
        dtype=np.float64,
    )

    return SurfaceMap(
        height_map=height_map,
        pixel_spacing=pixel_spacing,
        global_center=global_center,
    )


# Known MATLAB param key → ComparisonParams field mappings.
# All MATLAB length/size values are already in micrometers (not metres),
# except cellSizeX/Y which are in µm directly from param.pixSepX * cell_pixels.
_PARAM_MAP: dict[str, tuple[str, float]] = {
    # (python_field, scale_factor)
    # Cell size: MATLAB stores cellSizeX/cellSizeY in µm
    "cellSizeX": ("_cell_size_x", 1.0),  # handled specially below
    "cellSizeY": ("_cell_size_y", 1.0),  # handled specially below
    # Fill fraction: MATLAB stores as percentage (35 → 0.35)
    "cellFillRefMin": ("minimum_fill_fraction", 0.01),
    # Correlation threshold: dimensionless
    "cellSimMin": ("correlation_threshold", 1.0),
    # Angle threshold: degrees in both systems
    "cellAngleMax": ("angle_threshold", 1.0),
    # Position threshold: µm in both systems
    "cellDistMax": ("position_threshold", 1.0),
    # Search angle step: degrees
    "cellAngleStep": ("search_angle_step", 1.0),
}


def _build_comparison_params(param: dict) -> ComparisonParams:
    """
    Convert a MATLAB *param* dictionary to a :class:`ComparisonParams` instance.

    Any key not present in *param* retains the dataclass default value.

    :param param: Raw parameter dict loaded from ``metadata.json``.
    :returns: A :class:`ComparisonParams` ready for use.
    """
    kwargs: dict = {}
    for matlab_key, (py_field, scale) in _PARAM_MAP.items():
        if matlab_key not in param:
            continue
        value = param[matlab_key]
        if isinstance(value, list):
            kwargs[py_field] = np.array(value, dtype=np.float64) * scale
        else:
            kwargs[py_field] = float(value) * scale

    # Assemble cell_size from the separate X/Y keys
    if "_cell_size_x" in kwargs and "_cell_size_y" in kwargs:
        kwargs["cell_size"] = np.array(
            [kwargs.pop("_cell_size_x"), kwargs.pop("_cell_size_y")], dtype=np.float64
        )
    else:
        kwargs.pop("_cell_size_x", None)
        kwargs.pop("_cell_size_y", None)

    # search_angle_min mirrors search_angle_max (symmetric sweep around 0°)
    if "angle_threshold" in kwargs and "search_angle_max" not in kwargs:
        kwargs["search_angle_max"] = kwargs["angle_threshold"]
    if "search_angle_max" in kwargs and "search_angle_min" not in kwargs:
        kwargs["search_angle_min"] = -kwargs["search_angle_max"]

    return ComparisonParams(**kwargs)


def _run_pipeline(test_case: MatlabTestCase) -> ComparisonResult:
    """
    Build inputs and execute the full comparison pipeline for a test case.

    :param test_case: Loaded :class:`MatlabTestCase`.
    :returns: The pipeline's :class:`ComparisonResult`.
    """
    reference_map = _build_surface_map(test_case.data_con_pro_ref)
    comparison_map = _build_surface_map(test_case.data_con_pro_com)
    params = _build_comparison_params(test_case.param)
    return run_comparison_pipeline(reference_map, comparison_map, params)


# ---- Parametrized test fixtures ----

test_case_names = discover_test_cases()


@pytest.fixture(params=test_case_names, ids=test_case_names)
def test_case(request: pytest.FixtureRequest) -> MatlabTestCase:
    """Provide each test case as a fixture."""
    return load_test_case(request.param)


# ---- Comparison helpers ----


def assert_scalar_close(
    actual: float,
    expected: float,
    field_name: str,
    rtol: float = RTOL_SCALAR,
    atol: float = ATOL_SCALAR,
) -> None:
    """Assert two scalar values are close, handling NaN."""
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        assert actual is None or (isinstance(actual, float) and np.isnan(actual)), (
            f"{field_name}: expected NaN, got {actual}"
        )
        return

    if isinstance(expected, str):
        assert actual == expected, (
            f"{field_name}: expected '{expected}', got '{actual}'"
        )
        return

    assert np.isclose(actual, expected, rtol=rtol, atol=atol), (
        f"{field_name}: expected {expected}, got {actual}, "
        f"diff={abs(actual - expected):.2e}"
    )


# ---- Field accessors ----
# The MATLAB Res field names differ from the Python dataclass attribute names.
# Each field below is classified as one of:
#   MAPPED   – has a Python equivalent; accessor defined.
#   SKIPPED  – MATLAB-only bookkeeping with no Python equivalent; silently ignored.

# Fields that exist only in MATLAB and have no equivalent in the Python pipeline.
# They are silently ignored rather than failing the test.
_RESULT_FIELDS_SKIPPED: frozenset[str] = frozenset(
    {
        # --- Run-control booleans (always True in Python; not stored) ---
        "bArea",  # flag: area comparison step was executed
        "bCell",  # flag: cell comparison step was executed
        # --- File-path metadata – not part of the algorithm ---
        "pathReference",
        "pathCompare",
        # --- MATLAB-specific statistical test ---
        "bKM",  # Kolmogorov–Mirnov test flag; not implemented in Python
        # --- Pixel-dimension scalars from the raw containers (not algorithm outputs) ---
        "ds1",  # pixel size of reference container
        "ds2",  # pixel size of comparison container
        "ds",  # combined/representative pixel size
        # --- Scale ratio between the two datasets (not computed in Python) ---
        "dScale",
        # --- Roughness metrics (not computed in CMC-only pipeline) ---
        "sa12",  # Sa (mean-absolute) roughness
        "sq12",  # Sq (RMS) roughness
        "sqRef",
        "sqCom",
        # --- Global alignment quality metrics (not computed in CMC-only pipeline) ---
        "simVal",
        "accf",
        "ccf",
        "xcorr",
        "pOverlap",
        "lOverlap",
        "nOverlap",
        "angle1",
        "angle2",
        "dAngle",
        "cellAngle",
        "arccf",
        "consensusAngle",
        "consensusTx",
        "consensusTy",
    }
)

# MATLAB Res field name → Python ComparisonResult accessor.
_RESULT_FIELD_MAP: dict[str, callable] = {
    # ---- Cell counts ----
    "nCell": lambda r: len(r.cells),
    "nCmc": lambda r: r.congruent_matching_cells_count,
    "cmcVal": lambda r: r.congruent_matching_cells_count,
    "cmc": lambda r: r.congruent_matching_cells_count,
    # pCmc: fraction of processed cells that are CMC
    "pCmc": lambda r: r.congruent_matching_cells_count / len(r.cells)
    if r.cells
    else float("nan"),
    # pCmcArea: fraction of area covered by CMCs (equal-size cells → same as pCmc)
    "pCmcArea": lambda r: r.congruent_matching_cells_count / len(r.cells)
    if r.cells
    else float("nan"),
}


def _get_result_field(result: ComparisonResult, field_name: str):
    """
    Retrieve a named Res output field from a :class:`ComparisonResult`.

    Returns ``None`` for fields in ``_RESULT_FIELDS_SKIPPED`` (silently ignored).
    Raises ``KeyError`` for genuinely unknown fields.
    """
    if field_name in _RESULT_FIELDS_SKIPPED:
        return None

    if field_name in _RESULT_FIELD_MAP:
        return _RESULT_FIELD_MAP[field_name](result)

    if hasattr(result, field_name):
        return getattr(result, field_name)

    raise KeyError(f"Unknown result field: '{field_name}'")


# ---- Tests ----


@pytest.mark.skipif(
    not test_case_names,
    reason=f"No test cases found in {TEST_DATA_DIR}",
)
class TestCompareDatasetsNIST:
    """Test CompareDatasetsNIST Python implementation against MATLAB reference."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self, test_case: MatlabTestCase) -> None:
        """Run the pipeline once per test case and cache the result."""
        self.result = _run_pipeline(test_case)

    def test_res_scalar_fields(self, test_case: MatlabTestCase) -> None:
        """Test that Res scalar output fields match MATLAB."""
        comparable = {
            k: v
            for k, v in test_case.expected_res.items()
            if k not in _RESULT_FIELDS_SKIPPED
        }
        if not comparable:
            pytest.skip("No comparable scalar fields in Res output")

        failures = []
        for field_name, expected_val in comparable.items():
            try:
                actual_val = _get_result_field(self.result, field_name)
            except KeyError:
                failures.append(
                    f"  {field_name}: no mapping defined (expected {expected_val})"
                )
                continue

            try:
                assert_scalar_close(actual_val, expected_val, field_name)
            except AssertionError as exc:
                failures.append(f"  {exc}")

        if failures:
            pytest.fail("Scalar field mismatches:\n" + "\n".join(failures))


# ---- Standalone runner for quick checks ----

if __name__ == "__main__":
    """Quick check: load and print all test cases."""
    cases = discover_test_cases()
    if not cases:
        print(f"No test cases found in {TEST_DATA_DIR}")
        print("Run the MATLAB generator, then the conversion script first.")
    else:
        print(f"Found {len(cases)} test cases:\n")
        for name in cases:
            tc = load_test_case(name)
            print(f"  {name}: {len(tc.expected_res)} comparable scalar fields")
