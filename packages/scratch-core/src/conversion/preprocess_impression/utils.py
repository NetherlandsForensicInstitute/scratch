from numpy._typing import NDArray

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark


def update_mark_data(
    mark: Mark, data: NDArray, center: tuple[float, float] | None = None
) -> Mark:
    """
    Return a new Mark with updated scan data.

    :param mark: Original mark.
    :param data: New data array.
    :param center: New center, or None to recompute from data.
    :return: New Mark instance with updated data.
    """
    scan_image = mark.scan_image.model_copy(update={"data": data})
    return update_mark_scan_image(mark=mark, scan_image=scan_image, center=center)


def update_mark_scan_image(
    mark: Mark, scan_image: ScanImage, center: tuple[float, float] | None = None
) -> Mark:
    """
    Return a new Mark with updated scan image.

    :param mark: Original mark.
    :param scan_image: New scan image.
    :param center: New center, or None to recompute from data.
    :return: New Mark instance with updated scan image.
    """
    return mark.model_copy(update={"scan_image": scan_image, "_center": center})


Point2D = tuple[float, float]
