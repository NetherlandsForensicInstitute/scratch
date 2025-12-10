from image_generation.data_formats import ScanImage
from conversion.data_formats import MarkType, CropType, MarkImage


def scan_image_to_mark(scan_image: ScanImage, mark_type: MarkType, crop_type: CropType):
    return MarkImage(
        data=scan_image.data,
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
        mark_type=mark_type,
        crop_type=crop_type,
    )
