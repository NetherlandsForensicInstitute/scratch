import numpy as np
from lir.util import probability_to_logodds
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
    Transform CCF scores from [-1, +1] to log-odds using :func:`lir.util.probability_to_logodds`.

    CCF scores are first rescaled to probabilities in [0, 1] via ``y = (score + 1) / 2``,
    then converted to log10 log-odds. Boundary values (±1) are clipped by one ULP before
    rescaling to keep the result finite.

    :param scores: 1-D array of CCF scores in [-1, +1].
    :returns: 1-D array of log10 log-odds values in (-inf, +inf).
    """
    eps = np.finfo(float).eps
    clipped = np.clip(scores, -1 + eps, 1 - eps)
    y = (clipped + 1) / 2
    return probability_to_logodds(y)


def update_scan_image_data(scan_image: ScanImage, data: DepthData) -> ScanImage:
    """
    Return a new ScanImage with updated scan data.

    :param scan_image: Original scan_image.
    :param data: New data array.
    :return: New ScanImage instance with updated data.
    """
    return scan_image.model_copy(update={"data": data})
