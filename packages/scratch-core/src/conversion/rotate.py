from conversion.data_formats import CropType
from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


def rotate_image(
    scan_image: ScanImage,
    mask: MaskArray,
    param: dict,
    crop_type: CropType = CropType.RECTANGLE,
    rotation_angle: int = 0,
    crop_info: tuple = None,
) -> ScanImage:
    pass
