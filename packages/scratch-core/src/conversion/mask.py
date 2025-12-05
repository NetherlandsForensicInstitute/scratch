import numpy as np

from parsers import ScanImage
from utils.array_definitions import ScanMap2DArray, MaskArray


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
    data[mask == 0] = np.nan
    return data


def crop_to_mask(
    image: ScanMap2DArray,
    mask: MaskArray,
) -> ScanMap2DArray:
    """
    Crops an image to the bounding box of a mask.

    :param image: The image to crop
    :param mask: Binary mask
    :return: Cropped image containing only the masked region
    """
    y_slice, x_slice = _determine_bounding_box(mask)
    return image[y_slice, x_slice]


def _determine_bounding_box(mask: MaskArray) -> tuple[slice, slice]:
    """
    Determines the bounding box of non-zero values in a mask.

    :param mask: Binary mask array
    :return: Tuple of (y_slice, x_slice) for the bounding box
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return slice(x_min, x_max + 1), slice(y_min, y_max + 1)


def mask_and_crop_2d_array(
    image: ScanImage, mask: MaskArray, crop: bool = False
) -> ScanImage:
    """Apply the mask to the data and crop to the bounding box of the mask if `crop` is True."""
    data = image.data
    data = mask_2d_array(image=data, mask=mask)
    if crop:
        data = crop_to_mask(image=data, mask=mask)
    return ScanImage(
        data=data,
        scale_x=image.scale_x,
        scale_y=image.scale_y,
        meta_data=image.meta_data,
    )
