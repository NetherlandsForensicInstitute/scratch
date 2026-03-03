from pydantic import DirectoryPath, Field, FilePath, PositiveInt

from models import BaseModelConfig


class MarkDirectories(BaseModelConfig):
    mark_dir_ref: DirectoryPath
    mark_dir_comp: DirectoryPath

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return "SomethingWithNoValue"


class ImpressionParameters(BaseModelConfig): ...


class CalculateScoreImpression(MarkDirectories):
    param: ImpressionParameters


class StriationParamaters(BaseModelConfig):
    metadata_reference: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")
    metadata_compared: dict[str, str] = Field(..., description="fields needed for adding metadata to the plot")


class CalculateScoreStriation(MarkDirectories):
    param: StriationParamaters


class CalculateLR(MarkDirectories):
    lr_system_path: FilePath


class ImpressionLRParamaters(BaseModelConfig): ...


class CalculateLRImpression(CalculateLR):
    score: int
    n_cells: PositiveInt
    param: ImpressionLRParamaters


class StriationLRParamaters(BaseModelConfig): ...


class CalculateLRStriation(CalculateLR):
    score: float
    param: StriationLRParamaters
