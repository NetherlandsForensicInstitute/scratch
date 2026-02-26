"""
Integration test: CMC pipeline → build_impression_metrics → plot_impression_comparison_results.

Verifies that the full chain from raw surface data through the pipeline,
the metrics adapter, and the plot layer produces a valid RGB image without
errors, and that key fields in ImpressionComparisonMetrics are correctly
populated and converted from SI units to display units.
"""

import numpy as np

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.plots.impression_metrics import build_impression_metrics
from conversion.plots.plot_impression import plot_impression_comparison_results
from conversion.surface_comparison.models import ComparisonParams
from conversion.surface_comparison.pipeline import run_comparison_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCALE = 3.5e-6  # 3.5 µm/px — realistic breech face scan spacing


def _make_scan_image(data: np.ndarray, scale: float = _SCALE) -> ScanImage:
    return ScanImage(data=data, scale_x=scale, scale_y=scale)


def _make_mark(data: np.ndarray, scale: float = _SCALE) -> Mark:
    return Mark(
        scan_image=_make_scan_image(data, scale),
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
    )


def _aperiodic_surface(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Aperiodic surface with enough texture for ECC to register reliably."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols]
    data = (
        np.sin(x / 7.0) * np.cos(y / 11.0)
        + 0.3 * np.sin(x / 3.0 + y / 5.0)
        + 0.05 * rng.standard_normal((rows, cols))
    )
    return data.astype(np.float64)


def _default_params() -> ComparisonParams:
    return ComparisonParams(
        cell_size=np.array([3.5e-6 * 40, 3.5e-6 * 40]),  # 40 px cells
        search_angle_min=-2.0,
        search_angle_max=2.0,
        search_angle_step=1.0,
        correlation_threshold=0.3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_impression_metrics_cell_count():
    """Metrics grid shape matches expected rows/cols from image and cell size."""
    data = _aperiodic_surface(120, 120)
    scan = _make_scan_image(data)
    params = _default_params()

    result = run_comparison_pipeline(scan, scan, params)
    metrics = build_impression_metrics(result, params, scan)

    n_rows, n_cols = metrics.cell_correlations.shape
    assert n_rows > 0 and n_cols > 0
    assert metrics.has_cell_results is True


def test_build_impression_metrics_unit_conversion():
    """cell_positions_compared are in µm (not meters), cell_size_um is in µm."""
    data = _aperiodic_surface(120, 120)
    scan = _make_scan_image(data)
    params = _default_params()

    result = run_comparison_pipeline(scan, scan, params)
    metrics = build_impression_metrics(result, params, scan)

    # cell_size should be 40 px * 3.5 µm/px = 140 µm
    assert np.isclose(metrics.cell_size_um, 40 * 3.5, atol=0.1)

    # cell_positions_compared should be in µm range (image is ~420 µm wide)
    valid = metrics.cell_positions_compared[
        ~np.isnan(metrics.cell_positions_compared[:, 0])
    ]
    assert np.all(valid >= 0)
    assert np.all(valid < 1000)  # well within µm range, not meters (~0.00042)


def test_build_impression_metrics_max_error_conversion():
    """position_threshold and angle_threshold are correctly converted to display units."""
    data = _aperiodic_surface(120, 120)
    scan = _make_scan_image(data)
    params = _default_params()

    result = run_comparison_pipeline(scan, scan, params)
    metrics = build_impression_metrics(result, params, scan)

    # position_threshold default is 100e-6 m → 100 µm
    assert np.isclose(metrics.max_error_cell_position, 100.0, atol=0.01)
    # angle_threshold default is 2.0 degrees → unchanged
    assert np.isclose(metrics.max_error_cell_angle, 2.0, atol=0.01)


def test_build_impression_metrics_optional_fields_nan():
    """Fields not supplied by the pipeline default to NaN and set has_area_results=False."""
    data = _aperiodic_surface(120, 120)
    scan = _make_scan_image(data)
    params = _default_params()

    result = run_comparison_pipeline(scan, scan, params)
    metrics = build_impression_metrics(result, params, scan)

    assert np.isnan(metrics.area_correlation)
    assert metrics.has_area_results is False
    assert np.isnan(metrics.mean_square_ref)
    assert np.isnan(metrics.cutoff_low_pass)


def test_pipeline_to_plot_produces_rgb_image():
    """Full pipeline → metrics → plot produces a valid uint8 RGB array."""
    data = _aperiodic_surface(120, 120)
    scan = _make_scan_image(data)
    mark = _make_mark(data)
    params = _default_params()

    result = run_comparison_pipeline(scan, scan, params)
    metrics = build_impression_metrics(result, params, scan)

    plot = plot_impression_comparison_results(
        mark_reference_leveled=mark,
        mark_compared_leveled=mark,
        mark_reference_filtered=mark,
        mark_compared_filtered=mark,
        metrics=metrics,
        metadata_reference={"Sample": "Test reference", "Operator": "pytest"},
        metadata_compared={"Sample": "Test compared", "Operator": "pytest"},
    )

    overview = plot.comparison_overview
    assert overview.ndim == 3
    assert overview.shape[2] == 3
    assert overview.dtype == np.uint8
    assert overview.shape[0] > 0 and overview.shape[1] > 0
