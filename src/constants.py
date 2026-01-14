from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class RoutePrefix(StrEnum):
    COMPARATOR = "comparator"
    EXTRACTOR = "extractor"
    PREPROCESSOR = "preprocessor"
    PROCESSOR = "processor"
