import numpy as np
from matplotlib import pyplot as plt

import math


from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams, ComparisonResult
from conversion.plots.data_formats import ImpressionComparisonMetrics

import cv2
from typing import List, Tuple


def plot_rotated_squares(
    image: np.ndarray,
    squares: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
) -> np.ndarray:
    """
    Plots rotated squares on a float-based image.

    Args:
        image: Input array (e.g., magnitude 1e-6).
        squares: List of ((cx, cy), (w, h), angle_deg).

    Returns:
        Annotated image in BGR format (uint8).
    """
    # 1. Normalize the 1e-6 data to 0-255 range for visualization
    img_min, img_max = np.nanmin(image), np.nanmax(image)
    if img_max - img_min != 0:
        norm_img = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        norm_img = np.zeros(image.shape, dtype=np.uint8)

    # Convert grayscale to BGR to allow colored drawings
    color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

    # 2. Draw each rotated square
    for rect in squares:
        # cv2.boxPoints returns the 4 corners of the rotated rect
        box_points = cv2.boxPoints(rect)
        # Coordinates must be integers for drawing functions
        box_points = np.int64(box_points)

        # Draw the contour (Green, thickness 2)
        cv2.drawContours(color_img, [box_points], 0, (0, 255, 0), 2)
    return color_img


def plot_side_by_side(
    img1: np.ndarray, img2: np.ndarray, title1: str = "Image 1", title2: str = "Image 2"
) -> None:
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot first image
    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis("off")  # Hide grid/axes

    # Plot second image
    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


"""
Adapter that converts CMC pipeline outputs into ImpressionComparisonMetrics
for the plot layer.

This module is the single place where pipeline units (SI/meters) are
converted to display units (micrometers) expected by the plotting code.
"""


def build_impression_metrics(
    result: ComparisonResult,
    params: ComparisonParams,
    reference_map: ScanImage,
    area_correlation: float = float("nan"),
    mean_square_ref: float = float("nan"),
    mean_square_comp: float = float("nan"),
    mean_square_of_difference: float = float("nan"),
    cutoff_low_pass: float = float("nan"),
    cutoff_high_pass: float = float("nan"),
    cell_similarity_threshold: float = 0.25,
) -> ImpressionComparisonMetrics:
    """
    Convert a :class:`ComparisonResult` into an :class:`ImpressionComparisonMetrics`
    suitable for the plot layer.

    All pipeline spatial quantities are in meters; this function converts them
    to micrometers as required by the plotting code.

    The CMC pipeline stores results as a flat list of :class:`CellResult` objects
    in row-major order.  This function reshapes them into the 2-D grid that
    ``ImpressionComparisonMetrics`` expects, inferring the grid shape from the
    reference image dimensions and the cell size parameter.

    :param result: Output of :func:`run_comparison_pipeline`.
    :param params: The same :class:`ComparisonParams` used to run the pipeline.
    :param reference_map: The reference :class:`ScanImage` used to run the pipeline.
    :param area_correlation: Areal correlation coefficient from a separate
        area-based comparison step (NaN if not computed).
    :param mean_square_ref: Sq roughness of the reference surface in µm (NaN if not computed).
    :param mean_square_comp: Sq roughness of the compared surface in µm (NaN if not computed).
    :param mean_square_of_difference: Sq roughness of (comp − ref) in µm (NaN if not computed).
    :param cutoff_low_pass: Low-pass filter cutoff length in µm (NaN if not computed).
    :param cutoff_high_pass: High-pass filter cutoff length in µm (NaN if not computed).
    :param cell_similarity_threshold: Minimum ACCF score for a cell to be
        considered a CMC in the plots (default 0.25).
    :returns: :class:`ImpressionComparisonMetrics` ready for the plot layer.
    """
    cells = result.cells
    n_cells = len(cells)

    # --- Derive grid shape from reference image and cell size ---
    # physical_size and cell_size are both in meters.
    physical_size = reference_map.physical_size  # [width_m, height_m]
    cell_size_m = params.cell_size  # [cell_w_m, cell_h_m]
    n_cols = max(1, round(physical_size[0] / cell_size_m[0]))
    n_rows = max(1, round(physical_size[1] / cell_size_m[1]))

    # --- Build 2-D correlation grid (n_rows × n_cols), NaN for missing cells ---
    cell_correlations = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    cell_positions_compared = np.full((n_cells, 2), np.nan, dtype=np.float64)
    cell_rotations_compared = np.full(n_cells, np.nan, dtype=np.float64)

    for flat_idx, cell in enumerate(cells):
        row = flat_idx // n_cols
        col = flat_idx % n_cols
        if row < n_rows and col < n_cols:
            cell_correlations[row, col] = cell.best_score

        # Convert center_comparison from meters to µm for the plot layer
        cell_positions_compared[flat_idx] = [
            coord * 1e6 for coord in cell.center_comparison
        ]
        cell_rotations_compared[flat_idx] = cell.angle_deg

    # --- Scalar unit conversions ---
    cell_size_um = float(cell_size_m[0] * 1e6)  # assume square cells for display
    max_error_cell_position = float(params.position_threshold * 1e6)
    max_error_cell_angle = float(params.angle_deviation_threshold)
    cmc_area_fraction = result.cmc_area_fraction * 100  # fraction → percentage
    cmc_score = result.cmc_fraction * 100  # fraction → percentage

    return ImpressionComparisonMetrics(
        area_correlation=area_correlation,
        cell_correlations=cell_correlations,
        cmc_score=cmc_score,
        mean_square_ref=mean_square_ref,
        mean_square_comp=mean_square_comp,
        mean_square_of_difference=mean_square_of_difference,
        has_area_results=not math.isnan(area_correlation),
        has_cell_results=n_cells > 0,
        cell_positions_compared=cell_positions_compared,
        cell_rotations_compared=cell_rotations_compared,
        cmc_area_fraction=cmc_area_fraction,
        cutoff_low_pass=cutoff_low_pass,
        cutoff_high_pass=cutoff_high_pass,
        cell_size_um=cell_size_um,
        max_error_cell_position=max_error_cell_position,
        max_error_cell_angle=max_error_cell_angle,
        cell_similarity_threshold=cell_similarity_threshold,
    )
