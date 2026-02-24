from pydantic import DirectoryPath, FilePath, PositiveInt

from models import BaseModelConfig


class MarkDirectories(BaseModelConfig):
    mark_ref: DirectoryPath
    mark_comp: DirectoryPath


class ImpressionParameters(BaseModelConfig): ...


class CalculateScoreImpression(MarkDirectories):
    param: ImpressionParameters


class StriationParamaters(BaseModelConfig): ...


class CalculateScoreStriation(MarkDirectories):
    param: StriationParamaters


class CalculateLR(MarkDirectories):
    score: int
    lr_system: FilePath


class ImpressionLRParamaters(BaseModelConfig): ...


class CalculateLRImpression(CalculateLR):
    n_cells: PositiveInt
    param: ImpressionLRParamaters


class StriationLRParamaters(BaseModelConfig): ...


class CalculateLRStriation(CalculateLR):
    param: StriationLRParamaters
