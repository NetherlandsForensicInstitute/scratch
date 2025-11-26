from parsers.data_types import ScanImage


def subsample_data(scan_image: ScanImage, step_size: tuple[int, int]) -> ScanImage:
    """
    Subsample the data in a `ScanImage` instance by skipping `step_size` steps.

    :param scan_image: The instance of `ScanImage` containing the 2D image data to subsample.
    :param step_size: Denotes the number of steps to skip in each dimension. The first integer
        corresponds to the subsampling step size in the X-direction, and the second integer to
        the step size in the Y-direction.
    """
    step_x, step_y = step_size
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Step size must be a positive integer")
    if step_x >= scan_image.width or step_y >= scan_image.height:
        raise ValueError("Step size should be smaller than the image size")

    return ScanImage(
        data=scan_image.data[::step_y, ::step_x].copy(),
        scale_x=scan_image.scale_x * step_x,
        scale_y=scan_image.scale_y * step_y,
    )
