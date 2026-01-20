from enum import Enum, auto
from typing import Any

from pydantic import Field, computed_field

from container_models.base import ConfigBaseModel, BaseModel
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


class CropInfo(BaseModel):
    """
    Representation of the cropped area. Parameter `is_foreground` is used to indicate whether keep or delete the
    selected area.

    The points dict differs per CropType:
    CIRCLE: {'center': array [x, y], 'radius': float}
    RECTANGLE: {'corner': ScanMap2DArray}
    POLYGON: {'point': ScanMap2DArray}
    ELLIPSE: {'center': array [x, y], 'majoraxis': float, 'minoraxis': float, angle_majoraxis: float}
    """

    data: dict[str, Any]
    crop_type: CropType
    is_foreground: bool


class Mark(ConfigBaseModel):
    """
    Representation of a mark (impression or striation)
    """

    scan_image: ScanImage
    mark_type: MarkType
    crop_type: CropType
    meta_data: dict = Field(default_factory=dict)
    _center: tuple[float, float] | None = None

    @computed_field
    @property
    def center(self) -> tuple[float, float]:
        """
        Center point of the mark in image coordinates.

        Returns the center as (x, y) where x is the horizontal position
        (column) and y is the vertical position (row). If no explicit
        center has been set, computes it as the geometric center of the
        scan image.

        :returns: Center coordinates as (x, y),
        """
        if self._center is not None:
            return self._center
        data = self.scan_image.data
        return data.shape[1] / 2, data.shape[0] / 2
