import numpy as np

from computations.spatial import get_bounding_box
from container_models.base import BinaryMask, FloatArray2D
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


def mask_and_crop_2d_array(
    image: FloatArray2D, mask: BinaryMask, crop: bool = False
) -> FloatArray2D:
    """Apply the mask to the data and crop to the bounding box of the mask if `crop` is True."""
    image = mask_2d_array(image=image, mask=mask)
    if crop:
        image = crop_to_mask(image=image, mask=mask)
    return image


def mask_and_crop_scan_image(
    scan_image: ScanImage, mask: BinaryMask, crop: bool = False
) -> ScanImage:
    """Apply masking to the data in an instance of `ScanImage`."""
    masked_data = mask_and_crop_2d_array(image=scan_image.data, mask=mask, crop=crop)
    return ScanImage(
        data=masked_data, scale_x=scan_image.scale_x, scale_y=scan_image.scale_y
    )
