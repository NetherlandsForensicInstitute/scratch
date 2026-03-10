import numpy as np
from returns.io import IOResultE, IOSuccess
from returns.result import ResultE, Success

from container_models.scan_image import ScanImage
from container_models.base import DepthData


def unwrap_result[T](result: IOResultE[T] | ResultE[T]) -> T:
    match result:
        case IOSuccess(Success(value)) | Success(value):
            return value
        case _:
            assert False, "failed to unwrap"


def ccf_score_to_logodds(scores: np.ndarray) -> np.ndarray:
    """
    Transform CCF scores from [-1, +1] to [-inf, +inf] using a log10 logit.

    Rescales to [0, 1] then applies log-odds (base 10):
        y = (score + 1) / 2
        transformed = log10(y / (1 - y))

    Boundary values are clipped by one ULP to avoid infinite results.

    :param scores: 1-D array of raw CCF scores in [-1, +1].
    :returns: 1-D array of transformed scores.
    """
    eps = np.finfo(float).eps
    clipped = np.clip(scores, -1 + eps, 1 - eps)
    y = (clipped + 1) / 2
    return np.log10(y / (1 - y))


def update_scan_image_data(scan_image: ScanImage, data: DepthData) -> ScanImage:
    """
    Return a new ScanImage with updated scan data.

    :param scan_image: Original scan_image.
    :param data: New data array.
    :return: New ScanImage instance with updated data.
    """
    return scan_image.model_copy(update={"data": data})
