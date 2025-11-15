from enum import StrEnum, auto


class ImageType(StrEnum):
    SURFACE = auto()
    PROFILE = auto()
    IMAGE = auto()


class InputFormat(StrEnum):
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
