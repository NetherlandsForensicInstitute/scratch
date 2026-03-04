from conversion.data_formats import MarkMetadata
from pydantic import DirectoryPath, Field, FilePath, PositiveInt

from models import BaseModelConfig


class MarkDirectories(BaseModelConfig):
    mark_ref: DirectoryPath
    mark_comp: DirectoryPath

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return "SomethingWithNoValue"


class MetadataParameters(BaseModelConfig):
    metadata_reference: MarkMetadata = Field(..., description="Metadata identifying the reference mark.")
    metadata_compared: MarkMetadata = Field(..., description="Metadata identifying the compared mark.")


class CalculateScore(MarkDirectories):
    param: MetadataParameters


class CalculateLR(MarkDirectories):
    lr_system: FilePath
    param: MetadataParameters


class CalculateLRImpression(CalculateLR):
    score: int
    n_cells: PositiveInt


class CalculateLRStriation(CalculateLR):
    score: float
