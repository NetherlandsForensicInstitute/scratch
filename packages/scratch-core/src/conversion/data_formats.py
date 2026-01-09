from enum import Enum, auto
from typing import Any

from pydantic import BaseModel

from container_models.scan_image import ScanImage


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
    def scale(self) -> float:
        if self == MarkType.BREECH_FACE_IMPRESSION:
            return 3.5e-6
        return 1.5e-6


class CropType(Enum):
    RECTANGLE = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    POLYGON = auto()


# TODO: in java code kijken of er meer dan 1 cropinfo meegegeven kan worden -> ja, en alleen als 1e rectangle, dan
#  rotaten
# TODO: in java code de rotation angle default kijken (is die ooit anders door gebruiker?)


class CropInfo(BaseModel):
    """
    Representation of the cropped area. Parameter `is_foreground` is used to indicate whether keep or delete the
    selected area.
    TODO crop_type classes maken
    The points dict differs per CropType:
    CIRCLE: {'center': ScanMap2DArray, 'radius': float}
        RECTANGLE: {'corner': ScanMap2DArray}
        POLYGON: {'point': ScanMap2DArray}
        ELLIPSE: {'center': ScanMap2DArray, 'majoraxis': float, 'minoraxis': float, angle_majoraxis: float}
    """

    data: dict[str, Any]
    crop_type: CropType
    is_foreground: bool  # save crop or delete crop


class Mark(BaseModel):
    """
    Representation of a mark (impression or striation)
    """

    scan_image: ScanImage
    mark_type: MarkType
    crop_info: list[CropInfo]  # kan wss weggelaten worden
