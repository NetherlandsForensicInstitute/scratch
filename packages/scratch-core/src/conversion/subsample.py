from parsers.data_types import ScanImage, ScanDataKind


def subsample_data(
    scan_image: ScanImage, step_size: int | tuple[int, int]
) -> ScanImage:
    """
    Subsample the data in `ScanImage` instance by skipping `step_size` steps.

    If `step_size` is a tuple of integers, then it is assumed that the first integer
    corresponds to the subsampling step size in the X-direction, and the second
    integer to the Y-direction.

    :param scan_image: The instance of `ScanImage` to subsample.
    :param step_size: A single integer or a tuple of integers denoting the number of steps
        to skip in each dimension.
    """
    if isinstance(step_size, int):
        step_size = (step_size, step_size)

    step_x, step_y = step_size
    return ScanImage(
        data=scan_image.data[::step_y, ::step_x].copy(order="F"),
        scale_x=scan_image.scale_x * step_x,
        scale_y=scan_image.scale_y * step_y,
        path_to_original_image=scan_image.path_to_original_image,
        data_kind=ScanDataKind.SUBSAMPLED,
    )
