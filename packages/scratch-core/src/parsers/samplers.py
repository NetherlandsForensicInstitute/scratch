from skimage.transform import resize
from numpy.typing import NDArray
from container_models.scan_image import ScanImage
import numpy as np


def resample_scan_image(image: ScanImage, factors: tuple[float, float]) -> ScanImage:
    """
    Resample the ScanImage object using the specified resampling factors.

    :param image: Input ScanImage to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: The resampled ScanImage.
    """
    mask = (
        None
        if image.mask is None
        else _resample_image_array(image.mask, factors=factors)
    )
    return ScanImage(
        data=_resample_image_array(image.data, factors=factors),
        scale_x=image.scale_x * factors[0],
        scale_y=image.scale_y * factors[1],
        mask=mask,
    )


def _resample_image_array(
    array: NDArray,
    factors: tuple[float, float],
) -> NDArray:
    """
    Resample an array using the specified resampling factors.

    For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

    :param array: The array containing the image data to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: A numpy array containing the resampled image data.
    """
    factor_x, factor_y = factors
    resampled = resize(
        image=array,
        output_shape=(1 / factor_y * array.shape[0], 1 / factor_x * array.shape[1]),
        mode="edge",
        anti_aliasing=array.dtype != np.bool_ and all(factor > 1 for factor in factors),
    )
    return np.asarray(resampled, dtype=array.dtype)
