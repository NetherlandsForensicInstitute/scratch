import numpy as np
from matplotlib import pyplot as plt


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
