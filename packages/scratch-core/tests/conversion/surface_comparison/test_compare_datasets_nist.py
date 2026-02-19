"""Tests for CompareDatasetsNIST: compare Python implementation against MATLAB reference.

Usage:
    pytest test_compare_datasets_nist.py -v

Requires converted test cases in TEST_DATA_DIR (see convert_comparedatasetsnist_tests.py).
"""

import json
from dataclasses import dataclass, field
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
RTOL_ARRAY = 1e-5
ATOL_ARRAY = 1e-9


@dataclass
class DataContainer:
    """Minimal NFI data container for testing."""

    depth_data: np.ndarray
    xdim: float
    ydim: float
    mark_type: str = ""
    is_crop: int = 1
    is_prep: int = 1
    is_interp: int = 0
    data_param: dict = field(default_factory=dict)


@dataclass
class MatlabTestCase:
    """A single MATLAB test case with inputs and expected outputs."""

    case_name: str
    data_con_lev_ref: DataContainer
    data_con_lev_com: DataContainer
    data_con_pro_ref: DataContainer
    data_con_pro_com: DataContainer
    param: dict
    i_verbose: int
    expected_res: dict
    expected_res_arrays: dict[str, np.ndarray]
    expected_mapres_scalars: dict
    expected_mapres_arrays: dict[str, np.ndarray]

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

        case_name = meta["case_name"]

        # Load input containers
        containers = {}
        for cname in [
            "DataConLevRef",
            "DataConLevCom",
            "DataConProRef",
            "DataConProCom",
        ]:
            depth_path = case_dir / f"input_{cname}_depth_data.npy"
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing {depth_path}")

            depth_data = np.load(str(depth_path))
            xdim = meta.get(f"{cname}_xdim", 3.5e-6)
            ydim = meta.get(f"{cname}_ydim", 3.5e-6)
            mark_type = meta.get(f"{cname}_mark_type", "")
            is_crop = meta.get(f"{cname}_is_crop", 1)
            is_prep = meta.get(f"{cname}_is_prep", 1)
            is_interp = meta.get(f"{cname}_is_interp", 0)
            data_param = meta.get(f"{cname}_data_param", {})

            containers[cname] = DataContainer(
                depth_data=depth_data,
                xdim=float(xdim) if xdim is not None else 3.5e-6,
                ydim=float(ydim) if ydim is not None else 3.5e-6,
                mark_type=str(mark_type),
                is_crop=int(is_crop) if is_crop is not None else 1,
                is_prep=int(is_prep) if is_prep is not None else 1,
                is_interp=int(is_interp) if is_interp is not None else 0,
                data_param=data_param if isinstance(data_param, dict) else {},
            )

        # Load param
        param = meta.get("param", {})

        # Load iVerbose
        i_verbose = int(meta.get("iVerbose", 0))

        # Load expected Res output
        expected_res = _restore_nans(meta.get("output_Res", {}))

        # Load expected Res arrays
        expected_res_arrays = {}
        for field_name, npy_name in meta.get("output_Res_arrays", {}).items():
            arr_path = case_dir / npy_name
            if arr_path.exists():
                expected_res_arrays[field_name] = np.load(str(arr_path))

        # Load expected MapRes scalars
        expected_mapres_scalars = _restore_nans(meta.get("output_MapRes_scalars", {}))

        # Load expected MapRes arrays
        expected_mapres_arrays = {}
        for field_name, npy_name in meta.get("output_MapRes_arrays", {}).items():
            arr_path = case_dir / npy_name
            if arr_path.exists():
                expected_mapres_arrays[field_name] = np.load(str(arr_path))

        return cls(
            case_name=case_name,
            data_con_lev_ref=containers["DataConLevRef"],
            data_con_lev_com=containers["DataConLevCom"],
            data_con_pro_ref=containers["DataConProRef"],
            data_con_pro_com=containers["DataConProCom"],
            param=param,
            i_verbose=i_verbose,
            expected_res=expected_res,
            expected_res_arrays=expected_res_arrays,
            expected_mapres_scalars=expected_mapres_scalars,
            expected_mapres_arrays=expected_mapres_arrays,
        )


def _restore_nans(d: dict) -> dict:
    """Convert None values back to NaN for numeric fields."""
    result = {}
    for k, v in d.items():
        if v is None:
            result[k] = float("nan")
        elif isinstance(v, list):
            result[k] = [float("nan") if x is None else x for x in v]
        else:
            result[k] = v
    return result


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


def _build_surface_map(pro_con: DataContainer) -> SurfaceMap:
    """
    Construct a SurfaceMap from a processed MATLAB DataContainer.

    MATLAB pixel dimensions (xdim/ydim) are given in **metres**; they are
    converted to micrometres here so all downstream arithmetic is consistent.

    :param pro_con: Processed (filtered) DataContainer.
    :returns: A fully populated SurfaceMap.
    """
    pixel_spacing = np.array(
        [pro_con.xdim * _M_TO_UM, pro_con.ydim * _M_TO_UM], dtype=np.float64
    )

    height_map = pro_con.depth_data.astype(np.float64)

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


def _run_pipeline(
    test_case: MatlabTestCase,
) -> tuple[ComparisonResult, SurfaceMap]:
    """
    Build inputs and execute the full comparison pipeline for a test case.

    :param test_case: Loaded :class:`MatlabTestCase`.
    :returns: Tuple of (ComparisonResult, reference SurfaceMap).
              The reference map is returned so callers can derive pixel-count
              based metrics (e.g. nOverlap).
    """
    reference_map = _build_surface_map(test_case.data_con_pro_ref)
    comparison_map = _build_surface_map(test_case.data_con_pro_com)
    params = _build_comparison_params(test_case.param)
    result = run_comparison_pipeline(reference_map, comparison_map, params)
    return result, reference_map


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


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    field_name: str,
    rtol: float = RTOL_ARRAY,
    atol: float = ATOL_ARRAY,
) -> None:
    """Assert two arrays are close, handling NaN values."""
    assert actual.shape == expected.shape, (
        f"{field_name}: shape mismatch: {actual.shape} vs {expected.shape}"
    )

    # Compare NaN positions
    nan_actual = np.isnan(actual)
    nan_expected = np.isnan(expected)
    assert np.array_equal(nan_actual, nan_expected), (
        f"{field_name}: NaN positions differ. "
        f"actual has {nan_actual.sum()} NaNs, expected has {nan_expected.sum()}"
    )

    # Compare non-NaN values
    valid = ~nan_expected
    if valid.any():
        np.testing.assert_allclose(
            actual[valid],
            expected[valid],
            rtol=rtol,
            atol=atol,
            err_msg=f"{field_name}: values differ",
        )


# ---- Field accessors ----
# The MATLAB Res/MapRes field names differ from the Python dataclass attribute names.
# Each field below is classified as one of:
#   MAPPED   – has a Python equivalent; accessor defined.
#   SKIPPED  – MATLAB-only bookkeeping with no Python equivalent; silently ignored.

# Sentinel returned by _get_result_field for MATLAB-only fields that should be silently ignored.
_SKIP = object()

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
        # --- Roughness metrics ---
        # Sa (mean-absolute) roughness – Python only computes Sq (RMS); different metric.
        "sa12",
        # Sq roughness and global ACCF: the Python pipeline is CMC-only and does not
        # compute global alignment quality metrics.
        "sq12",
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
        # --- Res array outputs (MATLAB coordinate bookkeeping; no Python equivalent) ---
        "vCenterG1",  # global center of reference map [x,y]
        "vCenterL1",  # local center of reference map [x,y]
        "vPixSep1",  # pixel spacing of reference map [dx,dy]
        "vCenterG2",  # global center of comparison map [x,y]
        "vCenterL2",  # local center of comparison map [x,y]
        "vPixSep2",  # pixel spacing of comparison map [dx,dy]
        "vdPos",  # per-cell displacement vectors (raw registration output)
        "Par1",  # global alignment parameter vector for reference
        "Par2",  # global alignment parameter vector for comparison
        "Cell",  # 4×22 per-cell result matrix (MATLAB internal format)
        "vCellCenter",  # cell-grid center in global coords
        "vCellTrans",  # cell-grid translation vector
        "mPolCell",  # polygon vertices of cell shape (geometry metadata)
        # --- MapRes array outputs (raw aligned map data; no Python equivalent) ---
        "vm1",  # valid reference-map height values (1-D, indexed by vi)
        "vm2",  # valid comparison-map height values (1-D, aligned)
        "vi",  # flat linear indices of valid reference pixels
        "vCenterG",  # global center used for the alignment map [x,y]
        "vCenterL",  # local center of the alignment map [x,y]
        "vPixSep",  # pixel spacing of the alignment map [dx,dy]
        "vSize",  # pixel dimensions of the alignment map [cols,rows]
        "vAccf",  # per-pixel ACCF values over the valid region
    }
)


def _get_result_field(
    result: ComparisonResult, field_name: str, reference_map: SurfaceMap | None = None
):
    """
    Retrieve a named Res output field from a :class:`ComparisonResult`.

    Returns ``_SKIP`` sentinel for fields in ``_RESULT_FIELDS_SKIPPED``.
    Raises ``KeyError`` for genuinely unknown fields.

    MATLAB Res field → Python mapping
    -----------------------------------
    nCell           → total number of cells processed
    nCmc / cmcVal   → congruent_matching_cells_count
    pCmc            → nCmc / nCell  (fraction of cells that are CMC)
    pCmcArea        → nCmc / nCell  (equal-area cells → same as pCmc)
    """
    # Silently skip MATLAB-only bookkeeping fields
    if field_name in _RESULT_FIELDS_SKIPPED:
        return _SKIP

    _scalar_map: dict[str, callable] = {
        # ---- Cell counts ----
        "nCell": lambda r: len(r.cells),
        "nCmc": lambda r: r.congruent_matching_cells_count,
        "cmcVal": lambda r: r.congruent_matching_cells_count,
        "cmc": lambda r: r.congruent_matching_cells_count,
        # pCmc: fraction of processed cells that are CMC
        "pCmc": lambda r: r.congruent_matching_cells_count / len(r.cells)
        if r.cells
        else float("nan"),
        # pCmcArea: fraction of total area covered by CMCs (equal-size cells → same as pCmc)
        "pCmcArea": lambda r: r.congruent_matching_cells_count / len(r.cells)
        if r.cells
        else float("nan"),
    }

    if field_name in _scalar_map:
        return _scalar_map[field_name](result)

    # Fall back to direct attribute access on ComparisonResult
    if hasattr(result, field_name):
        return getattr(result, field_name)

    raise KeyError(f"Unknown result field: '{field_name}'")


def _get_mapres_field(result: ComparisonResult, field_name: str):
    """
    Retrieve a named MapRes field from a :class:`ComparisonResult`.

    MATLAB MapRes array field → Python mapping
    -------------------------------------------
    cellCentersRef  → Nx2 array of center_reference for all cells
    cellCentersCom  → Nx2 array of center_comparison for all cells
    cellAngles      → N-vector of registration_angle for all cells
    cellScores      → N-vector of area_cross_correlation_function_score
    cellFill        → N-vector of reference_fill_fraction
    isCMC           → N-vector of is_congruent (bool → int)
    """
    _array_map: dict[str, callable] = {
        "cellCentersRef": lambda r: np.array([c.center_reference for c in r.cells]),
        "cellCentersCom": lambda r: np.array([c.center_comparison for c in r.cells]),
        "cellAngles": lambda r: np.array([c.registration_angle for c in r.cells]),
        "cellScores": lambda r: np.array(
            [c.area_cross_correlation_function_score for c in r.cells]
        ),
        "cellFill": lambda r: np.array([c.reference_fill_fraction for c in r.cells]),
        "isCMC": lambda r: np.array([int(c.is_congruent) for c in r.cells]),
    }

    if field_name in _array_map:
        return _array_map[field_name](result)

    raise KeyError(f"Unknown MapRes field: '{field_name}'")


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
        self.result, self.reference_map = _run_pipeline(test_case)
        self.test_case = test_case

    def test_res_scalar_fields(self, test_case: MatlabTestCase) -> None:
        """Test that Res scalar output fields match MATLAB."""
        if not test_case.expected_res:
            pytest.skip("No scalar fields in Res output")

        failures = []
        for field_name, expected_val in test_case.expected_res.items():
            try:
                actual_val = _get_result_field(
                    self.result, field_name, self.reference_map
                )
            except KeyError:
                failures.append(
                    f"  {field_name}: no mapping defined (expected {expected_val})"
                )
                continue

            # Field is MATLAB-only with no Python equivalent — silently ignore
            if actual_val is _SKIP:
                continue

            try:
                assert_scalar_close(actual_val, expected_val, field_name)
            except AssertionError as exc:
                failures.append(f"  {exc}")

        if failures:
            pytest.fail("Scalar field mismatches:\n" + "\n".join(failures))

    def test_res_array_fields(self, test_case: MatlabTestCase) -> None:
        """Test that Res array output fields match MATLAB."""
        if not test_case.expected_res_arrays:
            pytest.skip("No array fields in Res output")

        failures = []
        for field_name, expected_arr in test_case.expected_res_arrays.items():
            try:
                actual_arr = _get_result_field(self.result, field_name)
            except KeyError:
                failures.append(f"  {field_name}: no mapping defined")
                continue

            try:
                assert_arrays_close(
                    np.asarray(actual_arr), expected_arr, f"Res.{field_name}"
                )
            except AssertionError as exc:
                failures.append(f"  {exc}")

        if failures:
            pytest.fail("Res array field mismatches:\n" + "\n".join(failures))

    def test_mapres_scalar_fields(self, test_case: MatlabTestCase) -> None:
        """Test that MapRes scalar output fields match MATLAB."""
        if not test_case.expected_mapres_scalars:
            pytest.skip("No scalar fields in MapRes output")

        failures = []
        for field_name, expected_val in test_case.expected_mapres_scalars.items():
            try:
                actual_val = _get_mapres_field(self.result, field_name)
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
            pytest.fail("MapRes scalar field mismatches:\n" + "\n".join(failures))

    def test_mapres_array_fields(self, test_case: MatlabTestCase) -> None:
        """Test that MapRes array output fields match MATLAB."""
        if not test_case.expected_mapres_arrays:
            pytest.skip("No array fields in MapRes output")

        failures = []
        for field_name, expected_arr in test_case.expected_mapres_arrays.items():
            try:
                actual_arr = _get_mapres_field(self.result, field_name)
            except KeyError:
                failures.append(f"  {field_name}: no mapping defined")
                continue

            try:
                assert_arrays_close(
                    np.asarray(actual_arr), expected_arr, f"MapRes.{field_name}"
                )
            except AssertionError as exc:
                failures.append(f"  {exc}")

        if failures:
            pytest.fail("MapRes array field mismatches:\n" + "\n".join(failures))


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
            n_res_scalars = len(tc.expected_res)
            n_res_arrays = len(tc.expected_res_arrays)
            n_mapres_scalars = len(tc.expected_mapres_scalars)
            n_mapres_arrays = len(tc.expected_mapres_arrays)
            print(
                f"  {name}: "
                f"Res({n_res_scalars} scalars, {n_res_arrays} arrays), "
                f"MapRes({n_mapres_scalars} scalars, {n_mapres_arrays} arrays)"
            )
            # Print a few key Res fields
            for k in ["ccf", "simVal", "xcorr", "pOverlap", "lOverlap"]:
                if k in tc.expected_res:
                    print(f"    {k}: {tc.expected_res[k]}")
