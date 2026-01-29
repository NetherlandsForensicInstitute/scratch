import numpy as np

from container_models.base import FloatArray2D, BinaryMask
from container_models.scan_image import ScanImage


def mask_2d_array(
    image: FloatArray2D,
    mask: BinaryMask,
) -> FloatArray2D:
    """
    Masks a 2D array by setting masked pixels to NaN.

    :param image: The image to mask
    :param mask: Binary mask where masked pixels = 0
    :return: Copy of image with masked pixels set to NaN
    """
    if mask.shape != image.shape:
        raise ValueError(f"Shape mismatch: image {image.shape} vs mask {mask.shape}")

    data = image.copy()
    data[~mask] = np.nan
    return data


def crop_to_mask(
    image: FloatArray2D,
    mask: BinaryMask,
    margin: int = 0,
) -> FloatArray2D:
    """
    Crops an image to the bounding box of a mask.

    :param image: The image to crop
    :param mask: Binary mask
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box
    :return: Cropped image containing only the masked region
    """
    x_slice, y_slice = get_bounding_box(mask, margin)
    return image[y_slice, x_slice]


def get_bounding_box(mask: BinaryMask, margin: int = 0) -> tuple[slice, slice]:
    """
    Determines the bounding box of non-zero values in a mask. If a margin is given, the bounding box will be expanded
    (in case of a negative margin) or cropped (in case of a positive margin) by `margin` pixels per side.

    :param mask: Binary mask array
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box
    :return: Tuple of (x_slice, y_slice) for the bounding box
    """
    non_zero_coords = np.nonzero(mask)
    if not non_zero_coords[0].size:
        raise ValueError("Mask is empty")

    y_coords, x_coords = np.nonzero(mask)
    y_min = max(0, y_coords.min() + margin)
    y_max = min(mask.shape[0], y_coords.max() - margin + 1)
    x_min = max(0, x_coords.min() + margin)
    x_max = min(mask.shape[1], x_coords.max() - margin + 1)

    if x_min >= x_max:
        raise ValueError("Slice results in x_min >= x_max. Margin may be too large.")
    if y_min >= y_max:
        raise ValueError("Slice results in y_min >= y_max. Margin may be too large.")

    return slice(x_min, x_max), slice(y_min, y_max)


def mask_and_crop_scan_image(
    scan_image: ScanImage, mask: BinaryMask, crop: bool = False
) -> ScanImage:
    """Apply masking to the data in an instance of `ScanImage`."""
    scan_image = scan_image.model_copy(
        update={"mask": mask}
    )  # QUICK FIX to get insynchrone with the transition from conversion
    scan_image.apply_mask_image()
    if crop:
        scan_image = scan_image.model_copy(
            update={"data": crop_to_mask(image=scan_image.data, mask=mask)}
        )
    return scan_image
