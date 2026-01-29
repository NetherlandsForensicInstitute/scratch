from container_models.base import Point
from loguru import logger


def clip_factors(factors: Point[float], preserve_aspect_ratio: bool) -> Point[float]:
    """Clip the scaling factors to minimum 1.0, while keeping the aspect ratio if `preserve_aspect_ratio` is True."""
    if preserve_aspect_ratio:
        max_factor = max(factors.x, factors.y)
        factors = Point(max_factor, max_factor)
        logger.debug(
            f"Preserving aspact ratio in clip factors, max point: {max_factor}"
        )
    return Point(max(factors.x, 1.0), max(factors.y, 1.0))
