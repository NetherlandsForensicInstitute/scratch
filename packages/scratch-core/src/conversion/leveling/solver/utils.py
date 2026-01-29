import numpy as np

from container_models.base import FloatArray
from container_models.scan_image import ScanImage


def compute_root_mean_square(data: FloatArray) -> float:
    """Compute the root-mean-square from a data array and return as Python float."""
    return float(np.sqrt(np.nanmean(data**2)))


def compute_image_center(scan_image: ScanImage) -> tuple[float, float]:
    """Compute the centerpoint (X, Y) of a scan image in physical coordinate space."""
    center_x = (scan_image.width - 1) * scan_image.scale_x * 0.5
    center_y = (scan_image.height - 1) * scan_image.scale_y * 0.5
    return center_x, center_y
