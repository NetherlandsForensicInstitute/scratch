from pydantic import DirectoryPath

from models import BaseModelConfig


class CalculateScore(BaseModelConfig):
    mark_dir_ref: DirectoryPath
    mark_dir_comp: DirectoryPath


class ImpressionParameters(BaseModelConfig): ...


class CalculateScoreImpression(CalculateScore):
    param: ImpressionParameters


class StriationParamaters(BaseModelConfig): ...


class CalculateScoreStriation(CalculateScore):
    param: StriationParamaters
