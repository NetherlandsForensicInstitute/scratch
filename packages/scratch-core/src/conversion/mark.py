from image_generation.data_formats import ScanImage
from conversion.data_formats import MarkType, CropType, MarkImage


def scan_image_to_mark(scan_image: ScanImage, mark_type: MarkType, crop_type: CropType):
    return MarkImage(
        scan_image=scan_image,
        mark_type=mark_type,
        crop_type=crop_type,
    )
