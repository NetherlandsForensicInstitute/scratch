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


def update_scan_image_data(scan_image: ScanImage, data: DepthData) -> ScanImage:
    """
    Return a new ScanImage with updated scan data.

    :param scan_image: Original scan_image.
    :param data: New data array.
    :return: New ScanImage instance with updated data.
    """
    return scan_image.model_copy(update={"data": data})
