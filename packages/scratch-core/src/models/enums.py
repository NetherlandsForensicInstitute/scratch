from enum import StrEnum, auto


# TODO: Rethink and rename this enum
class ImageType(StrEnum):
    SURFACE = auto()
    PROFILE = auto()
    IMAGE = auto()


class SupportedExtension(StrEnum):
    MAT = auto()
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    LMS = auto()
    PLU = auto()
    PNG = auto()
    BMP = auto()
    JPG = auto()
    JPEG = auto()


class CropType(StrEnum):
    RECTANGLE = auto()
    LINE = auto()
    POLYGON = auto()
    CIRCLE = auto()
