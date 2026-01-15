from typing import Optional

import numpy as np

from container_models.base import ScanMap2DArray, MaskArray
from container_models.scan_image import ScanImage


def mask_2d_array(
    image: ScanMap2DArray,
    mask: MaskArray,
) -> ScanMap2DArray:
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
    image: ScanMap2DArray,
    mask: MaskArray,
    margin: Optional[int] = None,
) -> ScanMap2DArray:
    """
    Crops an image to the bounding box of a mask.

    :param image: The image to crop
    :param mask: Binary mask
    :return: Cropped image containing only the masked region
    """
    x_slice, y_slice = _determine_bounding_box(mask, margin)
    return image[y_slice, x_slice]


def _determine_bounding_box(
    mask: MaskArray, margin: Optional[int] = None
) -> tuple[slice, slice]:
    """
    Determines the bounding box of non-zero values in a mask. If a margin is given, the bounding box will be expanded
    (in case of a negative margin) or cropped (in case of a positive margin) by with 2 * margin pixels.

    :param mask: Binary mask array
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box
    :return: Tuple of (y_slice, x_slice) for the bounding box
    """
    non_zero_coords = np.nonzero(mask)
    if not non_zero_coords[0].size:
        raise ValueError("Mask is empty")

    y_min, x_min = np.min(non_zero_coords, axis=1)
    y_max, x_max = np.max(non_zero_coords, axis=1)

    if margin:
        x_min = max(0, x_min + margin)
        x_max = min(mask.shape[1], x_max - margin + 1)
        y_min = max(0, y_min + margin)
        y_max = min(mask.shape[0], y_max - margin + 1)
    else:
        x_max += 1
        y_max += 1

    if x_min >= x_max:
        raise ValueError("Slice results in x_min >= x_max. Margin may be too large.")
    if y_min >= y_max:
        raise ValueError("Slice results in y_min >= y_max. Margin may be too large.")

    return slice(x_min, x_max), slice(y_min, y_max)


def mask_and_crop_2d_array(
    image: ScanMap2DArray, mask: MaskArray, crop: bool = False
) -> ScanMap2DArray:
    """Apply the mask to the data and crop to the bounding box of the mask if `crop` is True."""
    image = mask_2d_array(image=image, mask=mask)
    if crop:
        image = crop_to_mask(image=image, mask=mask)
    return image


def mask_and_crop_scan_image(
    scan_image: ScanImage, mask: MaskArray, crop: bool = False
) -> ScanImage:
    """Apply masking to the data in an instance of `ScanImage`."""
    masked_data = mask_and_crop_2d_array(image=scan_image.data, mask=mask, crop=crop)
    return ScanImage(
        data=masked_data, scale_x=scan_image.scale_x, scale_y=scan_image.scale_y
    )
