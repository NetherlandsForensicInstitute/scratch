from image_generation.translations import clip_data, normalize
from parsers.data_types import ScanImage
from utils.array_definitions import ScanMap2DArray


def get_array_for_display(image: ScanImage, std_scaler: float = 2.0) -> ScanMap2DArray:
    """
    Clip and normalize image data for displaying purposes.

    First the data will be clipped so that the values lie in the interval [μ - σ * S, μ + σ * S].
    Then the values are min-max normalized and scaled to the [0, 255] interval.

    :param image: An instance of `ScanImage`.
    :param std_scaler: The multiplier `S` for the standard deviation used above when clipping the image.
    :returns: An array containing the clipped and normalized image data.
    """
    clipped, lower, upper = clip_data(data=image.data, std_scaler=std_scaler)
    normalized = normalize(clipped, lower, upper)
    return normalized
