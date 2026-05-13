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
    CALCULATE_LR_IMPRESSION = "calculate-lr-impression"
    CALCULATE_LR_STRIATION = "calculate-lr-striation"


LIGHT_SOURCES = (
    LightSource(azimuth=90, elevation=45),
    LightSource(azimuth=180, elevation=45),
)
OBSERVER = LightSource(azimuth=90, elevation=45)


class LogLevel(StrEnum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class UrlFiles(StrEnum):
    def get_file_path(self, working_dir: Path) -> Path:
        """Return path to the file with the given working directory."""
        return working_dir / self.value

    def generate_url(self, access_url: str) -> str:
        """Generate the url to retrieve the file via the endpoint."""
        return f"{access_url}/{self.value}"
