from pydantic import DirectoryPath, FilePath, PositiveInt

from models import BaseModelConfig


class CalculateScore(BaseModelConfig):
    mark_ref: DirectoryPath
    mark_comp: DirectoryPath


class ImpressionParameters(BaseModelConfig): ...


class CalculateScoreImpression(CalculateScore):
    param: ImpressionParameters


class StriationParamaters(BaseModelConfig): ...


class CalculateScoreStriation(CalculateScore):
    param: StriationParamaters


class CalculateLR(CalculateScore):
    score: int
    lr_system: FilePath  # NOTE: is this a FilePath or DirectoryPath?


class ImpressionLRParamaters(BaseModelConfig): ...


class CalculateLRImpression(CalculateLR):
    n_cells: PositiveInt  # NOTE: this was an assumption (int)
    param: ImpressionLRParamaters


class StriationLRParamaters(BaseModelConfig): ...


class CalculateLRStriation(CalculateLR):
    param: StriationLRParamaters
