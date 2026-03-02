"""Tests for CompareDatasetsNIST: compare Python implementation against MATLAB reference.

Usage:
    pytest test_compare_datasets_nist.py -v

Requires converted test cases in TEST_DATA_DIR (see convert_comparedatasetsnist_tests.py).
"""

from pathlib import Path

import numpy as np
import pytest

from conversion.surface_comparison.models import ComparisonResult
from conversion.surface_comparison.pipeline import run_comparison_pipeline
from .helpers import MatlabTestCase, load_test_case

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
# ---- Constants ----

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "resources" / "cmc"

RTOL_SCALAR = 1e-6
ATOL_SCALAR = 1e-10

# ---- Helpers ----
"""
Visualize cell registration results in NIST CMC report style.

3-panel figure:
  Left:   Reference surface with cell grid (A1, A2 …)
  Centre: Comparison surface with registered cell outlines (B1, B2 …), rotated
  Right:  Cell ACCF distribution heatmap

Usage::

    from visualize_cells import plot_comparison_result

    def run_pipeline(test_case):
        result = run_comparison_pipeline(...)
        plot_comparison_result(
            result, test_case.reference_map, test_case.comparison_map,
            params=test_case.params, case_name=test_case.case_name,
        )
        return result
"""


@dataclass
class CellViz:
    """Per-cell data for plotting."""

    ref_x_um: float
    ref_y_um: float
    comp_x_um: float
    comp_y_um: float
    angle_deg: float
    score: float
    bbox_px: tuple[int, int, int, int] | None = None
    grid_row: int = 0
    grid_col: int = 0


def plot_registration(
    ref_data: np.ndarray,
    comp_data: np.ndarray,
    cells: list[CellViz],
    pixel_spacing: tuple[float, float],
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    NIST-style 3-panel visualization.

    Left:   reference surface + black dashed cell grid + A-labels
    Centre: comparison surface + rotated black dashed outlines + B-labels
    Right:  ACCF heatmap with scores
    """
    sx, sy = pixel_spacing
    um_x, um_y = sx * 1e6, sy * 1e6

    nrows_r, ncols_r = ref_data.shape
    nrows_c, ncols_c = comp_data.shape

    hw_r, hh_r = ncols_r * um_x / 2, nrows_r * um_y / 2
    ext_ref = [-hw_r, hw_r, -hh_r, hh_r]

    hw_c, hh_c = ncols_c * um_x / 2, nrows_c * um_y / 2
    ext_comp = [-hw_c, hw_c, -hh_c, hh_c]

    fig, (ax_r, ax_c, ax_h) = plt.subplots(1, 3, figsize=(20, 7))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # -- Surfaces --
    vlo, vhi = np.nanpercentile(ref_data, [2, 98])
    ax_r.imshow(
        ref_data,
        extent=ext_ref,
        cmap="inferno",
        origin="lower",
        vmin=vlo,
        vmax=vhi,
        aspect="equal",
        interpolation="nearest",
    )
    ax_r.set_title("Filtered Reference Surface A", fontsize=11)
    ax_r.set_xlabel("X - Position [µm]")
    ax_r.set_ylabel("Y - Position [µm]")

    vlo_c, vhi_c = np.nanpercentile(comp_data, [2, 98])
    ax_c.imshow(
        comp_data,
        extent=ext_comp,
        cmap="inferno",
        origin="lower",
        vmin=vlo_c,
        vmax=vhi_c,
        aspect="equal",
        interpolation="nearest",
    )
    ax_c.set_title("Filtered Compared Surface B", fontsize=11)
    ax_c.set_xlabel("X - Position [µm]")
    ax_c.set_ylabel("Y - Position [µm]")

    if not cells:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig

    n_grid_rows = max(c.grid_row for c in cells) + 1
    n_grid_cols = max(c.grid_col for c in cells) + 1
    score_grid = np.full((n_grid_rows, n_grid_cols), np.nan)

    for c in cells:
        if c.bbox_px is None:
            continue

        r0, c0, r1, c1 = c.bbox_px
        # Pixel bounds → centred µm
        x0 = c0 * um_x - hw_r
        x1 = c1 * um_x - hw_r
        y0 = hh_r - r1 * um_y
        y1 = hh_r - r0 * um_y
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        cell_num = c.grid_row * n_grid_cols + c.grid_col + 1

        # --- Reference: black dashed rectangle ---
        ax_r.add_patch(
            patches.Rectangle(
                (x0, y0),
                w,
                h,
                lw=1.5,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
            )
        )

        ax_r.text(
            cx,
            cy,
            f"A{cell_num}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

        # --- Comparison: rotated rectangle at registered position ---
        if np.isfinite(c.comp_x_um) and np.isfinite(c.comp_y_um):
            comp_cx = c.comp_x_um
            comp_cy = c.comp_y_um
            angle = c.angle_deg

            # FancyBboxPatch centred at comp position, rotated
            rect_c = patches.Rectangle(
                (-w / 2, -h / 2),
                w,
                h,
                lw=1.5,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
            )

            # Apply rotation around centre, then translate to position
            t = (
                mtransforms.Affine2D().rotate_deg(angle).translate(comp_cx, comp_cy)
                + ax_c.transData
            )
            rect_c.set_transform(t)
            ax_c.add_patch(rect_c)

            ax_c.text(
                comp_cx,
                comp_cy,
                f"B{cell_num}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
            )

        # Score grid
        score_grid[c.grid_row, c.grid_col] = c.score

    # --- ACCF heatmap ---
    all_x0 = min(c.bbox_px[1] * um_x - hw_r for c in cells if c.bbox_px)
    all_x1 = max(c.bbox_px[3] * um_x - hw_r for c in cells if c.bbox_px)
    all_y0 = min(hh_r - c.bbox_px[2] * um_y for c in cells if c.bbox_px)
    all_y1 = max(hh_r - c.bbox_px[0] * um_y for c in cells if c.bbox_px)
    hm_extent = [all_x0, all_x1, all_y0, all_y1]

    im = ax_h.imshow(
        np.ma.masked_invalid(score_grid),
        extent=hm_extent,
        origin="lower",
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )

    # Grid lines and labels on heatmap
    for c in cells:
        if c.bbox_px is None:
            continue
        r0, c0, r1, c1 = c.bbox_px
        x0 = c0 * um_x - hw_r
        x1 = c1 * um_x - hw_r
        y0 = hh_r - r1 * um_y
        y1 = hh_r - r0 * um_y
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        ax_h.add_patch(
            patches.Rectangle(
                (x0, y0), w, h, lw=1.0, edgecolor="black", facecolor="none"
            )
        )

        cell_num = c.grid_row * n_grid_cols + c.grid_col + 1
        score_str = f"{c.score:.2f}" if np.isfinite(c.score) else "NaN"

        ax_h.text(
            cx,
            cy + h * 0.15,
            f"A{cell_num}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )
        ax_h.text(
            cx,
            cy - h * 0.15,
            score_str,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )

    ax_h.set_title("Cell ACCF Distribution", fontsize=11)
    ax_h.set_xlabel("X - Position [µm]")
    ax_h.set_ylabel("Y - Position [µm]")
    fig.colorbar(im, ax=ax_h, shrink=0.7, pad=0.02).set_label("ACCF [-]")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# =====================================================================
# High-level: ComparisonResult / SurfaceMap / CellResult
# =====================================================================


def plot_comparison_result(
    result,
    reference_map,
    comparison_map,
    params=None,
    case_name: str = "",
    out_dir: str = "cell_viz",
) -> str | None:
    """
    Produce a NIST-style PNG from pipeline output.

    :param result: ``ComparisonResult`` with ``.cells``.
    :param reference_map: Reference ``SurfaceMap``.
    :param comparison_map: Comparison ``SurfaceMap``.
    :param params: ``ComparisonParams`` (for cell size).
    :param case_name: Used in title and filename.
    :param out_dir: Output directory.
    :returns: Path to saved PNG.
    """
    if not result.cells:
        return None

    sx = reference_map.scale_x
    sy = reference_map.scale_y
    um_x, um_y = sx * 1e6, sy * 1e6
    nrows, ncols = reference_map.data.shape
    hw = ncols * um_x / 2
    hh = nrows * um_y / 2

    cell_w_px, cell_h_px = None, None
    if params is not None:
        cell_w_px = int(round(params.cell_size[0] / sx))
        cell_h_px = int(round(params.cell_size[1] / sy))

    # Sort cells into grid by reference position
    ref_positions = [
        (c.center_reference[0] * 1e6, c.center_reference[1] * 1e6) for c in result.cells
    ]
    xs = sorted(set(round(p[0], 1) for p in ref_positions))
    ys = sorted(set(round(p[1], 1) for p in ref_positions))

    def _idx(val, vals, tol=5.0):
        for i, v in enumerate(vals):
            if abs(val - v) < tol:
                return i
        return 0

    viz_cells = []
    for cell in result.cells:
        ref_x_m, ref_y_m = cell.center_reference
        comp_x_m, comp_y_m = cell.center_comparison
        accf = cell.area_cross_correlation_function_score
        angle = cell.registration_angle

        ref_x_um = ref_x_m * 1e6
        ref_y_um = ref_y_m * 1e6

        grid_col = _idx(round(ref_x_um, 1), xs)
        grid_row = _idx(round(ref_y_um, 1), ys)

        bbox = None
        if cell_w_px is not None and cell_h_px is not None:
            center_col = ref_x_m / sx
            center_row = ref_y_m / sy
            r0 = max(0, int(round(center_row - cell_h_px / 2)))
            c0 = max(0, int(round(center_col - cell_w_px / 2)))
            r1 = min(r0 + cell_h_px, nrows)
            c1 = min(c0 + cell_w_px, ncols)
            bbox = (r0, c0, r1, c1)

        viz_cells.append(
            CellViz(
                ref_x_um=ref_x_um - hw,
                ref_y_um=hh - ref_y_um,
                comp_x_um=comp_x_m * 1e6 - hw if np.isfinite(comp_x_m) else np.nan,
                comp_y_um=hh - comp_y_m * 1e6 if np.isfinite(comp_y_m) else np.nan,
                angle_deg=np.degrees(angle) if np.isfinite(angle) else 0.0,
                score=accf,
                bbox_px=bbox,
                grid_row=grid_row,
                grid_col=grid_col,
            )
        )

    n_cells = len(result.cells)
    n_cmc = getattr(result, "congruent_matching_cells_count", "?")
    title = f"{case_name}  —  {n_cells} cells, {n_cmc} CMC"

    Path(out_dir).mkdir(exist_ok=True)
    save_path = str(Path(out_dir) / f"{case_name or 'result'}.png")

    fig = plot_registration(
        reference_map.data,
        comparison_map.data,
        viz_cells,
        pixel_spacing=(sx, sy),
        title=title,
        save_path=save_path,
    )
    plt.close(fig)
    return save_path


def discover_test_cases() -> list[str]:
    """Find all available test case directories."""
    if not TEST_DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in TEST_DATA_DIR.iterdir() if (d / "metadata.json").exists()
    )


def run_pipeline(test_case: MatlabTestCase) -> ComparisonResult:
    result = run_comparison_pipeline(
        test_case.reference_map,
        test_case.reference_map_processed,
        test_case.comparison_map,
        test_case.comparison_map_processed,
        test_case.params,
    )
    plot_comparison_result(
        result,
        test_case.reference_map,
        test_case.comparison_map,
        params=test_case.params,
        case_name=test_case.case_name,
    )
    return result


# ---- Fixtures ----

test_case_names = discover_test_cases()


@pytest.fixture(params=test_case_names, ids=test_case_names)
def test_case(request: pytest.FixtureRequest) -> MatlabTestCase:
    return load_test_case(TEST_DATA_DIR / request.param)


# ---- Tests ----


@pytest.mark.skipif(
    not test_case_names, reason=f"No test cases found in {TEST_DATA_DIR}"
)
class TestCompareDatasetsNIST:
    """Test CompareDatasetsNIST Python implementation against MATLAB reference."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self, test_case: MatlabTestCase) -> None:
        self.result = run_pipeline(test_case)

    def test_res_scalar_fields(self, test_case: MatlabTestCase) -> None:
        """Test that Res scalar output fields match MATLAB."""
        if not test_case.expected_results:
            pytest.skip("No comparable scalar fields in Res output")

        failures = []
        for field, expected in test_case.expected_results.items():
            actual = getattr(self.result, field)

            if np.isnan(expected):
                if not np.isnan(actual):
                    failures.append(f"  {field}: expected NaN, got {actual}")
            elif not np.isclose(actual, expected, rtol=RTOL_SCALAR, atol=ATOL_SCALAR):
                failures.append(
                    f"  {field}: expected {expected}, got {actual}, diff={abs(actual - expected):.2e}"
                )

        if failures:
            pytest.fail("Scalar field mismatches:\n" + "\n".join(failures))
