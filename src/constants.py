from enum import StrEnum
from pathlib import Path

from container_models.light_source import LightSource

PROJECT_ROOT = Path(__file__).parent.parent


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
    CALCULATE_SCORE_IMPRESSION = "calculate-score-impression"
    CALCULATE_SCORE_STRIATION = "calculate-score-striation"


class ComparatorEndpoint(StrEnum):
    ROOT = ""


LIGHT_SOURCES = (
    LightSource(azimuth=90, elevation=45),
    LightSource(azimuth=180, elevation=45),
)
OBSERVER = LightSource(azimuth=90, elevation=45)
