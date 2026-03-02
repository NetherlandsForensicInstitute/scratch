"""
Test cell_registration against MATLAB reference data.

Loads cell_corr_test_data.json, constructs ScanImage objects, calls
register_cells, and compares per-cell results against MATLAB's vPos2,
dAngle, and ACCF.

Usage:
    python new_test_fine_reg.py [cell_corr_test_data.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration import register_cells
from conversion.surface_comparison.models import ComparisonParams

POS_TOL_M = 5e-6
ANGLE_TOL_RAD = 0.01
ACCF_TOL = 0.05


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

    # vPixSep is [row_sep, col_sep] → scale_y, scale_x for ScanImage.
    vPixSep1 = np.array(inp["map1_vPixSep"])
    vPixSep2 = np.array(inp["map2_vPixSep"])
    reference_image = ScanImage(
        data=np.array(inp["map1_data"], dtype=np.float64),
        scale_x=float(vPixSep1[1]),
        scale_y=float(vPixSep1[0]),
    )
    comparison_image = ScanImage(
        data=np.array(inp["map2_data"], dtype=np.float64),
        scale_x=float(vPixSep2[1]),
        scale_y=float(vPixSep2[0]),
    )

    # vCellSize is [row_size, col_size]; ComparisonParams.cell_size is [x, y].
    vCellSize = np.array(inp["vCellSize"])
    params = ComparisonParams(
        cell_size=np.array([vCellSize[1], vCellSize[0]]),
        minimum_fill_fraction=inp["cellFillRefMin"],
        search_angle_min=np.degrees(inp["shiftAngleMin"]),
        search_angle_max=np.degrees(inp["shiftAngleMax"]),
    )

    if verbose:
        print(
            f"  Map1: {reference_image.data.shape}, "
            f"scale=[{vPixSep1[0] * 1e6:.2f}, {vPixSep1[1] * 1e6:.2f}] µm/px"
        )
        print(
            f"  Map2: {comparison_image.data.shape}, "
            f"scale=[{vPixSep2[0] * 1e6:.2f}, {vPixSep2[1] * 1e6:.2f}] µm/px"
        )
        print(f"  cellSize: {vCellSize * 1e6} µm")

    results_list = register_cells(reference_image, comparison_image, params)

    if verbose:
        print(
            f"\n  Python produced {len(results_list)} cells, MATLAB has {len(matlab_cells)}"
        )

    is_different_mark = name == "different_mark"
    test_results = {"name": name, "n_cells_matlab": tc["nCells"], "cell_results": []}

    for i, mc in enumerate(matlab_cells):
        matlab_vpos1 = np.array(mc["vPos1"])  # [row, col] in m
        matlab_vpos2 = np.array(mc["vPos2"])  # [row, col] in m
        matlab_dangle = mc["dAngle"]  # radians
        matlab_accf = mc["accf"]

        # Match Python cell by closest center_reference.
        # Cell.center_reference is [x, y]; MATLAB vPos1 is [row, col] = [y, x].
        best_match = None
        best_dist = np.inf
        for cell in results_list:
            cell_rc = np.array([cell.center_reference[1], cell.center_reference[0]])
            d = np.linalg.norm(cell_rc - matlab_vpos1)
            if d < best_dist:
                best_dist = d
                best_match = cell

        if best_match is None or best_dist > 50e-6:
            if verbose:
                print(
                    f"  Cell {i}: NO MATCH "
                    f"(vPos1=[{matlab_vpos1[0] * 1e6:.1f}, {matlab_vpos1[1] * 1e6:.1f}])"
                )
            test_results["cell_results"].append({"matlab_idx": i, "status": "NO_MATCH"})
            continue

        cell = best_match

        if (
            cell.center_comparison is None
            or cell.angle_reference is None
            or cell.best_score is None
        ):
            if verbose:
                print(f"  Cell {i}: REGISTRATION FAILED")
            test_results["cell_results"].append({"matlab_idx": i, "status": "REG_FAIL"})
            continue

        # Convert Cell [x, y] back to [row, col] for position comparison.
        python_vpos2 = np.array([cell.center_comparison[1], cell.center_comparison[0]])

        pos2_err = np.linalg.norm(python_vpos2 - matlab_vpos2)
        angle_err = abs(np.radians(cell.angle_reference) - matlab_dangle)
        if angle_err > np.pi:
            angle_err = 2 * np.pi - angle_err
        accf_err = abs(cell.best_score - matlab_accf)

        if is_different_mark:
            both_low = matlab_accf < 0.2 and abs(cell.best_score) < 0.2
            cell_pass = both_low
        else:
            pos2_ok = pos2_err < POS_TOL_M
            angle_ok = angle_err < ANGLE_TOL_RAD
            accf_ok = accf_err < ACCF_TOL
            cell_pass = pos2_ok and angle_ok and accf_ok

        cell_result = {
            "matlab_idx": i,
            "matlab_vPos2_um": (matlab_vpos2 * 1e6).tolist(),
            "python_vPos2_um": (python_vpos2 * 1e6).tolist(),
            "matlab_dAngle_deg": np.degrees(matlab_dangle),
            "python_dAngle_deg": cell.angle_reference,
            "matlab_accf": matlab_accf,
            "python_accf": cell.best_score,
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
                f"  P:pos2=[{python_vpos2[0] * 1e6:.1f},{python_vpos2[1] * 1e6:.1f}]"
                f"  M:ang={np.degrees(matlab_dangle):.3f}°"
                f"  P:ang={cell.angle_reference:.3f}°"
                f"  M:accf={matlab_accf:.4f}"
                f"  P:accf={cell.best_score:.4f}"
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
