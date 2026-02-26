import numpy as np
from returns.io import IOResultE, IOSuccess
from returns.result import ResultE, Success

from container_models.scan_image import ScanImage
from container_models.base import DepthData, FloatArray


def unwrap_result[T](result: IOResultE[T] | ResultE[T]) -> T:
    match result:
        case IOSuccess(Success(value)) | Success(value):
            return value
        case _:
            assert False, "failed to unwrap"


def update_scan_image_data(scan_image: ScanImage, data: DepthData) -> ScanImage:
    """
    Return a new ScanImage with updated scan data.

    :param scan_image: Original scan_image.
    :param data: New data array.
    :return: New ScanImage instance with updated data.
    """
    return scan_image.model_copy(update={"data": data})


def compute_roughness_sa(data: FloatArray) -> float:
    """
    Compute arithmetic mean roughness (ISO 25178 Sa parameter) of a profile.

    Sa is the arithmetic mean of the absolute values of the profile heights,
    calculated as: mean(|z|). The 'S' denotes a surface/areal parameter and
    'a' denotes arithmetical mean.

    :param data: profile array. May contain NaN values which are ignored.
    :returns: Arithmetic mean roughness (Sa) in the same units as the input profile.
    """
    return float(np.nanmean(np.abs(data - np.nanmean(data))))


def compute_roughness_sqrt(data: FloatArray) -> float:
    """
    Compute root-mean-square roughness (ISO 25178 Sq parameter) of a profile.

    Sq is the root-mean-square (or standard deviation with ddof=0) of the profile heights, calculated as:
    sqrt(mean(z^2)). The 'S' denotes a surface/areal parameter and 'q'
    denotes quadratic mean (root-mean-square).

    :param data: profile array. May contain NaN values which are ignored.
    :returns: Root-mean-square roughness (Sq) in the same units as the input profile.
    """
    return float(np.sqrt(np.nanmean((data - np.nanmean(data)) ** 2)))
