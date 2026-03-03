from pydantic import DirectoryPath, Field, FilePath, PositiveInt

from models import BaseModelConfig


class MarkDirectories(BaseModelConfig):
    mark_ref: DirectoryPath
    mark_comp: DirectoryPath

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return "SomethingWithNoValue"


class ImpressionParameters(BaseModelConfig): ...


class CalculateScoreImpression(MarkDirectories):
    param: ImpressionParameters


class StriationParameters(BaseModelConfig):
    metadata_reference: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")
    metadata_compared: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")


class CalculateScoreStriation(MarkDirectories):
    param: StriationParameters


class CalculateLR(MarkDirectories):
    score: int
    lr_system: FilePath


class ImpressionLRParameters(BaseModelConfig): ...


class CalculateLRImpression(CalculateLR):
    n_cells: PositiveInt
    param: ImpressionLRParameters


class StriationLRParameters(BaseModelConfig): ...


class CalculateLRStriation(CalculateLR):
    param: StriationLRParameters
