from enum import StrEnum
from pathlib import Path

from container_models.light_source import LightSource
from conversion.data_formats import MarkType

PROJECT_ROOT = Path(__file__).parent.parent


class StriationMarks(StrEnum):
    APERTURE_SHEAR = "aperture shear striation mark"
    BULLET_GEA = "bullet gea striation mark"
    BULLET_LEA = "bullet lea striation mark"
    CHAMBER = "chamber striation mark"
    EJECTOR = "ejector striation mark"
    EJECTOR_PORT = "ejector port striation mark"
    EXTRACTOR = "extractor striation mark"
    FIRING_PIN_DRAG = "firing pin drag striation mark"


class ImpressionMarks(StrEnum):
    BREECH_FACE = "breech face impression mark"
    CHAMBER = "chamber impression mark"
    EJECTOR = "ejector impression mark"
    EXTRACTOR = "extractor impression mark"
    FIRING_PIN = "firing pin impression mark"


IMPRESSION_TO_MARK_TYPE: dict[ImpressionMarks, MarkType] = {
    ImpressionMarks.BREECH_FACE: MarkType.BREECH_FACE_IMPRESSION,
    ImpressionMarks.CHAMBER: MarkType.CHAMBER_IMPRESSION,
    ImpressionMarks.EJECTOR: MarkType.EJECTOR_IMPRESSION,
    ImpressionMarks.EXTRACTOR: MarkType.EXTRACTOR_IMPRESSION,
    ImpressionMarks.FIRING_PIN: MarkType.FIRING_PIN_IMPRESSION,
}

STRIATION_TO_MARK_TYPE: dict[StriationMarks, MarkType] = {
    StriationMarks.APERTURE_SHEAR: MarkType.APERTURE_SHEAR_STRIATION,
    StriationMarks.BULLET_GEA: MarkType.BULLET_GEA_STRIATION,
    StriationMarks.BULLET_LEA: MarkType.BULLET_LEA_STRIATION,
    StriationMarks.CHAMBER: MarkType.CHAMBER_STRIATION,
    StriationMarks.EJECTOR: MarkType.EJECTOR_STRIATION,
    StriationMarks.EJECTOR_PORT: MarkType.EJECTOR_PORT_STRIATION,
    StriationMarks.EXTRACTOR: MarkType.EXTRACTOR_STRIATION,
    StriationMarks.FIRING_PIN_DRAG: MarkType.FIRING_PIN_DRAG_STRIATION,
}


class MaskTypes(StrEnum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"


class RoutePrefix(StrEnum):
    COMPARATOR = "comparator"
    EXTRACTOR = "extractor"
    PREPROCESSOR = "preprocessor"
    PROCESSOR = "processor"


class PreprocessorEndpoint(StrEnum):
    ROOT = ""
    PROCESS_SCAN = "process-scan"
    PREPARE_MARK_IMPRESSION = "prepare-mark-impression"
    PREPARE_MARK_STRIATION = "prepare-mark-striation"
    EDIT_SCAN = "edit-scan"


class ExtractorEndpoint(StrEnum):
    ROOT = ""
    FILES = "files/{token}/{filename}"


class ProcessorEndpoint(StrEnum):
    ROOT = ""


class ComparatorEndpoint(StrEnum):
    ROOT = ""


LIGHT_SOURCES = (
    LightSource(azimuth=90, elevation=45),
    LightSource(azimuth=180, elevation=45),
)
OBSERVER = LightSource(azimuth=90, elevation=45)
