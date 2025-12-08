from utils.array_definitions import ScanMap2DArray


def subsample_array(
    scan_data: ScanMap2DArray, step_size: tuple[int, int]
) -> ScanMap2DArray:
    """
    Subsample the data in a `ScanImage` instance by skipping `step_size` steps.

    :param scan_image: The instance of `ScanImage` containing the 2D image data to subsample.
    :param step_size: Denotes the number of steps to skip in each dimension. The first integer
        corresponds to the subsampling step size in the X-direction, and the second integer to
        the step size in the Y-direction.

    """
    step_x, step_y = step_size
    width, height = scan_data.shape
    if step_x >= width or step_y >= height:
        raise ValueError("Step size should be smaller than the image size")
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Step size must be a tuple of positive integers")

    return scan_data[::step_y, ::step_x].copy()
