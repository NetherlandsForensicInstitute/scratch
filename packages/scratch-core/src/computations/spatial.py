import numpy as np
from loguru import logger
from skimage.transform import resize

from container_models.base import BinaryMask, FloatArray2D
from container_models.scan_image import ScanImage


def get_bounding_box(mask: BinaryMask, margin: int) -> tuple[slice, slice]:
    """
    Compute the minimal bounding box of a 2D mask.

    Finds the smallest axis-aligned rectangle containing all non-zero (or True) values.

    :param mask: 2D mask (non-zero/True values indicate the region of interest)
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box
    :returns: Tuple (y_slice, x_slice) as slices for bounding_box.
    """
    y_coords, x_coords = np.nonzero(mask)
    y_min = max(0, y_coords.min() + margin)
    y_max = min(mask.shape[0], y_coords.max() - margin + 1)
    x_min = max(0, x_coords.min() + margin)
    x_max = min(mask.shape[1], x_coords.max() - margin + 1)

    if x_min >= x_max:
        raise ValueError("Slice results in x_min >= x_max. Margin may be too large.")
    if y_min >= y_max:
        raise ValueError("Slice results in y_min >= y_max. Margin may be too large.")

    return slice(y_min, y_max), slice(x_min, x_max)


def make_isotropic(scan_image: ScanImage) -> ScanImage:
    """
    Resample a scan image to isotropic resolution (i.e. equal pixel spacing in X and Y).

    If the scan image is already isotropic, the original instance is returned.
    Otherwise, the image data is upsampled to the highest available resolution
    (the smaller of the two scale factors) using nearest-neighbor interpolation.
    Note: NaN values are preserved and will not be interpolated.

    :param scan_image: The ScanImage instance to be resampled.
    :returns: A new ScanImage instance with isotropic scaling.
    """
    if _is_isotropic(scan_image):
        return scan_image

    # Upsample to the smallest pixel scale (highest resolution)
    target_scale = min(scan_image.scale_x, scan_image.scale_y)
    upsampled = _upsample_image_data(scan_image, target_scale)

    return ScanImage(
        data=upsampled,
        scale_x=target_scale,
        scale_y=target_scale,
        meta_data=scan_image.meta_data,
    )


def subsample_scan_image(
    scan_image: ScanImage, step_size_x: int, step_size_y: int
) -> ScanImage:
    """
    Subsample the data in a `ScanImage` instance by skipping steps in each dimension.
    :param scan_image: The instance of `ScanImage` containing the 2D image data to subsample.
    :param step_size_x: The number of steps to skip in the X-direction.
    :param step_size_y: The number of steps to skip in the Y-direction.
    :returns: A subsampled `ScanImage` with updated scales. If both step sizes are 1, the original
        `ScanImage` instance is returned.
    """
    if step_size_x == 1 and step_size_y == 1:
        logger.info("No subsampling needed, returning original scan image")
        return scan_image

    width, height = scan_image.width, scan_image.height
    if not (0 < step_size_x < width and 0 < step_size_y < height):
        raise ValueError(
            f"Step size should be positive and smaller than the image size: {(height, width)}"
        )
    logger.info(
        f"Subsampling scan image with step sizes x: {step_size_x}, y: {step_size_y}"
    )
    return ScanImage(
        data=scan_image.data[::step_size_y, ::step_size_x].copy(),
        scale_x=scan_image.scale_x * step_size_x,
        scale_y=scan_image.scale_y * step_size_y,
    )


def _is_isotropic(scan_image: ScanImage) -> bool:
    """Check if a scan image is isotropic within tolerance."""
    tolerance = 1e-16
    return bool(np.isclose(scan_image.scale_x, scan_image.scale_y, atol=tolerance))


def _get_target_shape(scan_image: ScanImage, target_scale: float) -> tuple[int, int]:
    """Get the target shape for a scan image given a target scale."""
    height, width = (
        int(round(scan_image.height * scan_image.scale_y / target_scale)),
        int(round(scan_image.width * scan_image.scale_x / target_scale)),
    )
    return height, width


def _upsample_image_data(scan_image: ScanImage, target_scale: float) -> FloatArray2D:
    """Upsample image data in a `ScanImage` instance to a common target scale."""
    upsampled = resize(
        image=scan_image.data,
        output_shape=_get_target_shape(scan_image, target_scale),
        mode="edge",
        anti_aliasing=False,  # Disabled for pure upsampling
        preserve_range=True,  # Keep original data intensity levels
        order=0,  # Nearest Neighbor so that NaNs appear at corresponding coordinates
    )
    return np.asarray(upsampled, dtype=np.float64)
