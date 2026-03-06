"""
Tests for ``cell_corr_analysis`` against MATLAB reference data.

Loads ``ecc_test_data.json``, constructs ``MapStruct`` objects with
MATLAB's exact coordinate metadata, calls ``cell_corr_analysis``, and
compares per-cell results against MATLAB's ``vPos2``, ``dAngle``, and ACCF.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from conversion.surface_comparison.cell_registration import (
    MapStruct,
    cell_corr_analysis,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

POS_TOL_M = 5e-6  # m
ANGLE_TOL_RAD = 0.01  # rad
ACCF_TOL = 0.05
ACCF_LOW_TOL = 0.15  # for "different mark" cases where both scores must be low

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TEST_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_PATH = TEST_ROOT / "resources" / "cmc" / "ecc" / "ecc_test_data.json"


def _load_test_cases() -> list[dict]:
    with open(TEST_DATA_PATH) as f:
        return json.load(f)


_TEST_CASES = _load_test_cases()
_TEST_IDS = [tc["name"] for tc in _TEST_CASES]


@pytest.fixture(params=_TEST_CASES, ids=_TEST_IDS)
def matlab_test_case(request) -> dict:
    return request.param


# ---------------------------------------------------------------------------
# Helper: run cell_corr_analysis for one test case
# ---------------------------------------------------------------------------


def _run_case(tc: dict) -> tuple[list, list]:
    """Return (results_list, matlab_cells) for a test case dict."""
    inp = tc["inputs"]
    matlab_cells = tc["cells"]

    map1 = MapStruct(
        map=np.array(inp["map1_data"], dtype=np.float64),
        vCenterG=np.array(inp["map1_vCenterG"]),
        vCenterL=np.array(inp["map1_vCenterL"]),
        angle=inp["map1_angle"],
        vPixSep=np.array(inp["map1_vPixSep"]),
    )
    map2 = MapStruct(
        map=np.array(inp["map2_data"], dtype=np.float64),
        vCenterG=np.array(inp["map2_vCenterG"]),
        vCenterL=np.array(inp["map2_vCenterL"]),
        angle=inp["map2_angle"],
        vPixSep=np.array(inp["map2_vPixSep"]),
    )

    vCellPosition = (
        np.array(matlab_cells[0]["vPos1"]) if matlab_cells else map1.vCenterL.copy()
    )

    results_list = cell_corr_analysis(
        map1=map1,
        map2=map2,
        vCellSize=np.array(inp["vCellSize"]),
        vCellPosition=vCellPosition,
        shiftAngleMin=inp["shiftAngleMin"],
        shiftAngleMax=inp["shiftAngleMax"],
        cellFillRefMin=inp["cellFillRefMin"],
        cellFillRegMin=0.25,
        cellFillRedMax=0.50,
        viParLevel=np.array([1]),
        verbose=False,
    )

    return results_list, matlab_cells


def _find_match(results_list, matlab_vpos1: np.ndarray):
    """Return the Python result closest to matlab_vpos1, or None if too far."""
    best, best_dist = None, np.inf
    for pr in results_list:
        d = np.linalg.norm(pr.vPos1 - matlab_vpos1)
        if d < best_dist:
            best_dist, best = d, pr
    return best if (best is not None and best_dist <= 50e-6) else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cell_count(matlab_test_case: dict) -> None:
    """Python produces the same number of cells as MATLAB."""
    results_list, matlab_cells = _run_case(matlab_test_case)
    assert len(results_list) == len(matlab_cells), (
        f"[{matlab_test_case['name']}] cell count: "
        f"expected {len(matlab_cells)}, got {len(results_list)}"
    )


def test_cell_position(matlab_test_case: dict) -> None:
    """vPos2 for every cell matches the MATLAB reference within POS_TOL_M."""
    if matlab_test_case["name"] == "different_mark":
        pytest.skip("different_mark: position comparison not meaningful")

    results_list, matlab_cells = _run_case(matlab_test_case)

    failures = []
    for i, mc in enumerate(matlab_cells):
        pr = _find_match(results_list, np.array(mc["vPos1"]))
        if pr is None:
            failures.append(f"  cell {i}: no match found")
            continue
        if not pr.bValid:
            failures.append(f"  cell {i}: registration failed")
            continue
        err = np.linalg.norm(pr.vPos2 - np.array(mc["vPos2"]))
        if err >= POS_TOL_M:
            failures.append(
                f"  cell {i}: pos2 error {err * 1e6:.2f} µm "
                f"(tol {POS_TOL_M * 1e6:.0f} µm)"
            )

    if failures:
        pytest.fail(
            f"[{matlab_test_case['name']}] position mismatches:\n" + "\n".join(failures)
        )


def test_cell_angle(matlab_test_case: dict) -> None:
    """dAngle for every cell matches the MATLAB reference within ANGLE_TOL_RAD."""
    if matlab_test_case["name"] == "different_mark":
        pytest.skip("different_mark: angle comparison not meaningful")

    results_list, matlab_cells = _run_case(matlab_test_case)

    failures = []
    for i, mc in enumerate(matlab_cells):
        pr = _find_match(results_list, np.array(mc["vPos1"]))
        if pr is None or not pr.bValid:
            continue
        err = abs(pr.dAngle - mc["dAngle"])
        if err > np.pi:
            err = 2 * np.pi - err
        if err >= ANGLE_TOL_RAD:
            failures.append(
                f"  cell {i}: angle error {np.degrees(err):.3f}° "
                f"(tol {np.degrees(ANGLE_TOL_RAD):.3f}°)"
            )

    if failures:
        pytest.fail(
            f"[{matlab_test_case['name']}] angle mismatches:\n" + "\n".join(failures)
        )


def test_cell_accf(matlab_test_case: dict) -> None:
    """ACCF for every cell matches MATLAB within ACCF_TOL; for 'different_mark'
    both Python and MATLAB scores must be below ACCF_LOW_TOL."""
    results_list, matlab_cells = _run_case(matlab_test_case)
    is_different_mark = matlab_test_case["name"] == "different_mark"

    failures = []
    for i, mc in enumerate(matlab_cells):
        pr = _find_match(results_list, np.array(mc["vPos1"]))
        if pr is None or not pr.bValid:
            continue
        if is_different_mark:
            if not (mc["accf"] < ACCF_LOW_TOL and abs(pr.accf) < ACCF_LOW_TOL):
                failures.append(
                    f"  cell {i}: expected both ACCF < {ACCF_LOW_TOL}, "
                    f"got matlab={mc['accf']:.4f} python={pr.accf:.4f}"
                )
        else:
            err = abs(pr.accf - mc["accf"])
            if err >= ACCF_TOL:
                failures.append(
                    f"  cell {i}: ACCF error {err:.4f} "
                    f"(matlab={mc['accf']:.4f} python={pr.accf:.4f}, tol {ACCF_TOL})"
                )

    if failures:
        pytest.fail(
            f"[{matlab_test_case['name']}] ACCF mismatches:\n" + "\n".join(failures)
        )
