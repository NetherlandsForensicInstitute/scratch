"""
Test the new cell_registration_matlab.py against MATLAB reference data.

Loads cell_corr_test_data.json, constructs MapStruct objects with MATLAB's
exact coordinate metadata, calls cell_corr_analysis, and compares per-cell
results against MATLAB's vPos2, dAngle, and ACCF.

Usage:
    python test_new_registration.py [cell_corr_test_data.json]
"""

import json
import sys
from pathlib import Path
import numpy as np

from conversion.surface_comparison_simone.cell_registration import (
    MapStruct,
    cell_corr_analysis,
)

# Tolerances
POS_TOL_M = 5e-6
ANGLE_TOL_RAD = 0.01
ACCF_TOL = 0.05
ACCF_LOW_TOL = 0.15


def load_test_data(json_path: str) -> list[dict]:
    with open(json_path) as f:
        return json.load(f)


def run_test_case(tc: dict, verbose: bool = True) -> dict:
    name = tc["name"]
    inp = tc["inputs"]
    matlab_cells = tc["cells"]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Test: {name}")
        print(f"  {tc['description']}")
        print(f"  MATLAB cells: {tc['nCells']}")

    # Build MapStruct objects with MATLAB's exact metadata
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

    vCellSize = np.array(inp["vCellSize"])

    if verbose:
        print(
            f"  Map1: {map1.map.shape}, vCG={map1.vCenterG * 1e6}, angle={np.degrees(map1.angle):.2f}°"
        )
        print(
            f"  Map2: {map2.map.shape}, vCG={map2.vCenterG * 1e6}, angle={np.degrees(map2.angle):.2f}°"
        )
        print(f"  cellSize: {vCellSize * 1e6} µm")

    # Use MATLAB vPos1 of first cell as seed position
    # (In real usage this comes from cell_position_optim)
    if matlab_cells:
        vCellPosition = np.array(matlab_cells[0]["vPos1"])
    else:
        vCellPosition = map1.vCenterL.copy()

    # Run cell correlation analysis
    results_list = cell_corr_analysis(
        map1=map1,
        map2=map2,
        vCellSize=vCellSize,
        vCellPosition=vCellPosition,
        shiftAngleMin=inp["shiftAngleMin"],
        shiftAngleMax=inp["shiftAngleMax"],
        cellFillRefMin=inp["cellFillRefMin"],
        cellFillRegMin=0.25,
        cellFillRedMax=0.50,
        viParLevel=np.array([1]),
        verbose=verbose,
    )

    if verbose:
        print(
            f"\n  Python produced {len(results_list)} cells, MATLAB has {len(matlab_cells)}"
        )

    # Match Python cells to MATLAB cells by vPos1 proximity
    is_different_mark = name == "different_mark"
    test_results = {"name": name, "n_cells_matlab": tc["nCells"], "cell_results": []}

    for i, mc in enumerate(matlab_cells):
        matlab_vpos1 = np.array(mc["vPos1"])
        matlab_vpos2 = np.array(mc["vPos2"])
        matlab_dangle = mc["dAngle"]
        matlab_accf = mc["accf"]

        # Find matching Python cell by closest vPos1
        best_match = None
        best_dist = np.inf
        for pr in results_list:
            d = np.linalg.norm(pr.vPos1 - matlab_vpos1)
            if d < best_dist:
                best_dist = d
                best_match = pr

        if best_match is None or best_dist > 50e-6:
            if verbose:
                print(
                    f"  Cell {i}: NO MATCH (vPos1=[{matlab_vpos1[0] * 1e6:.1f}, {matlab_vpos1[1] * 1e6:.1f}])"
                )
            test_results["cell_results"].append({"matlab_idx": i, "status": "NO_MATCH"})
            continue

        pr = best_match

        if not pr.bValid:
            if verbose:
                print(f"  Cell {i}: REGISTRATION FAILED")
            test_results["cell_results"].append({"matlab_idx": i, "status": "REG_FAIL"})
            continue

        # Compare
        pos2_err = np.linalg.norm(pr.vPos2 - matlab_vpos2)
        angle_err = abs(pr.dAngle - matlab_dangle)
        if angle_err > np.pi:
            angle_err = 2 * np.pi - angle_err
        accf_err = abs(pr.accf - matlab_accf)

        if is_different_mark:
            # For different marks, both ACCF should be low
            both_low = matlab_accf < 0.2 and abs(pr.accf) < 0.2
            cell_pass = both_low
        else:
            accf_tol = ACCF_TOL
            pos2_ok = pos2_err < POS_TOL_M
            angle_ok = angle_err < ANGLE_TOL_RAD
            accf_ok = accf_err < accf_tol
            cell_pass = pos2_ok and angle_ok and accf_ok

        cell_result = {
            "matlab_idx": i,
            "matlab_vPos2_um": (matlab_vpos2 * 1e6).tolist(),
            "python_vPos2_um": (pr.vPos2 * 1e6).tolist(),
            "matlab_dAngle_deg": np.degrees(matlab_dangle),
            "python_dAngle_deg": np.degrees(pr.dAngle),
            "matlab_accf": matlab_accf,
            "python_accf": pr.accf,
            "err_pos2_um": pos2_err * 1e6,
            "err_angle_deg": np.degrees(angle_err),
            "err_accf": accf_err,
            "pass": cell_pass,
            "status": "PASS" if cell_pass else "FAIL",
        }
        test_results["cell_results"].append(cell_result)

        if verbose:
            status = "PASS" if cell_pass else "FAIL"
            flags = []
            if not is_different_mark:
                if pos2_err >= POS_TOL_M:
                    flags.append(f"pos2Δ={pos2_err * 1e6:.2f}µm")
                if angle_err >= ANGLE_TOL_RAD:
                    flags.append(f"angΔ={np.degrees(angle_err):.3f}°")
                if accf_err >= ACCF_TOL:
                    flags.append(f"accfΔ={accf_err:.4f}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(
                f"  Cell {i}: {status}{flag_str}"
                f"  M:pos2=[{matlab_vpos2[0] * 1e6:.1f},{matlab_vpos2[1] * 1e6:.1f}]"
                f"  P:pos2=[{pr.vPos2[0] * 1e6:.1f},{pr.vPos2[1] * 1e6:.1f}]"
                f"  M:ang={np.degrees(matlab_dangle):.3f}°"
                f"  P:ang={np.degrees(pr.dAngle):.3f}°"
                f"  M:accf={matlab_accf:.4f}"
                f"  P:accf={pr.accf:.4f}"
            )

    n_pass = sum(1 for cr in test_results["cell_results"] if cr.get("status") == "PASS")
    n_total = len(test_results["cell_results"])
    test_results["status"] = "PASS" if (n_pass == n_total and n_total > 0) else "FAIL"
    if verbose:
        print(f"\n  Result: {test_results['status']} ({n_pass}/{n_total} cells)")
    return test_results


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "cell_corr_test_data.json"
    if not Path(json_path).exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)

    test_cases = load_test_data(json_path)
    print(f"Loaded {len(test_cases)} test cases from {json_path}")

    all_results = []
    for tc in test_cases:
        all_results.append(run_test_case(tc, verbose=True))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    n_pass = 0
    for r in all_results:
        cells = r.get("cell_results", [])
        n_cp = sum(1 for cr in cells if cr.get("status") == "PASS")
        status = r.get("status", "ERROR")
        if status == "PASS":
            n_pass += 1
        print(f"  {status:5s}  {r['name']} ({n_cp}/{len(cells)} cells)")

    print(f"\nOverall: {n_pass}/{len(all_results)} test cases passed")
    sys.exit(0 if n_pass == len(all_results) else 1)


if __name__ == "__main__":
    main()
