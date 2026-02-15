from models import BaseModelConfig


class CalculateScore(BaseModelConfig):
    mark_dir_ref: str
    mark_dir_comp: str


class ImpressionParameters(BaseModelConfig): ...


class CalculateScoreImpression(CalculateScore):
    param: ImpressionParameters


class StriationParamaters(BaseModelConfig): ...


class CalculateScoreStriation(CalculateScore):
    param: StriationParamaters
