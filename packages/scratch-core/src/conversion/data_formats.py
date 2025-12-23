from enum import Enum, auto

from pydantic import BaseModel

from image_generation.data_formats import ScanImage


class MarkType(Enum):
    # Impression marks
    BREECH_FACE_IMPRESSION = auto()
    CHAMBER_IMPRESSION = auto()
    EJECTOR_IMPRESSION = auto()
    EXTRACTOR_IMPRESSION = auto()
    FIRING_PIN_IMPRESSION = auto()

    # Striation marks
    APERTURE_SHEAR_STRIATION = auto()
    BULLET_GEA_STRIATION = auto()
    BULLET_LEA_STRIATION = auto()
    CHAMBER_STRIATION = auto()
    EJECTOR_STRIATION = auto()
    EJECTOR_PORT_STRIATION = auto()
    EXTRACTOR_STRIATION = auto()
    FIRING_PIN_DRAG_STRIATION = auto()

    def is_impression(self) -> bool:
        return "IMPRESSION" in self.name

    def is_striation(self) -> bool:
        return "STRIATION" in self.name

    @property
    def sampling_rate(self) -> float:
        if self == MarkType.BREECH_FACE_IMPRESSION:
            return 3.5e-6
        return 1.5e-6


class CropType(Enum):
    RECTANGLE = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    POLYGON = auto()


class MarkImage(BaseModel):
    """
    Representation of a mark (impression or striation)
    """

    scan_image: ScanImage
    mark_type: MarkType
    crop_type: CropType
