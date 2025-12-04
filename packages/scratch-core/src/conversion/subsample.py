from image_generation.translations import ScanMap2DArray


def subsample_data(
    scan_image: ScanMap2DArray, step_size_x: int, step_size_y: int
) -> ScanMap2DArray:
    """
    Subsample the data in a `ScanMap2D` instance by skipping `step_size` steps.

    :param scan_image: The instance of `ScanMap2D` containing the 2D image data to subsample.
    :param step_size_x: Denotes the number of steps to skip in X-direction.
    :param step_size_y: Denotes the number of steps to skip in Y-direction.
    """
    width, height = scan_image.shape
    if not (0 < step_size_x < width and 0 < step_size_y < height):
        raise ValueError("Step size should be positive and smaller than the image size")

    return scan_image[::step_size_y, ::step_size_x].copy()
