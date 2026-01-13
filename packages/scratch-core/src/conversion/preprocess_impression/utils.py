from numpy._typing import NDArray

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark


def update_mark_data(mark: Mark, data: NDArray) -> Mark:
    """
    Return a new Mark with updated scan data.

    :param mark: Original mark.
    :param data: New data array.
    :return: New Mark instance with updated data.
    """
    scan_image = mark.scan_image.model_copy(update={"data": data})
    return update_mark_scan_image(mark, scan_image)


def update_mark_scan_image(mark: Mark, scan_image: ScanImage) -> Mark:
    """
    Return a new Mark with updated scan image.

    :param mark: Original mark.
    :param scan_image: New scan image.
    :return: New Mark instance with updated scan image.
    """
    return mark.model_copy(update={"scan_image": scan_image})


Point2D = tuple[float, float]
