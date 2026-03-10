
from dataclasses import dataclass

from conversion.data_formats import Mark


@dataclass(frozen=True)
class ProcessedMark:
    filtered_mark: Mark
    leveled_mark: Mark


def compare_surfaces()