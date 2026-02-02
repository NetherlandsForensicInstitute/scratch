from enum import StrEnum
from pathlib import Path

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
    BREACH_FACE = "breach face impression mark"
    CHAMBER = "chamber impression mark"
    EJECTOR = "ejector impression mark"
    EXTRACTOR = "extractor impression mark"
    FIRING_PIN = "firing pin impression mark"


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
