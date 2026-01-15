import numpy as np

from conversion.data_formats import Mark
from conversion.preprocess_impression.utils import update_mark_scan_image
from conversion.resample import (
    resample_scan_image_and_mask,
)


def resample(mark: Mark, target_scale: float) -> Mark:
    """
    Resample mark if target scale differs from current scale.

    :param mark: Input mark.
    :param target_scale: Target pixel scale.
    :return: resampled mark
    """
    resampled, _ = resample_scan_image_and_mask(
        mark.scan_image,
        target_scale=target_scale,
        only_downsample=False,
    )
    return update_mark_scan_image(mark, resampled)


def needs_resampling(mark: Mark, target_scale: float) -> bool:
    """
    Check if mark needs resampling to target scale.

    :param mark: Input mark.
    :param target_scale: Target pixel scale.
    :return: True if resampling is needed.
    """
    current_scale = (mark.scan_image.scale_x, mark.scan_image.scale_y)
    return not np.allclose(current_scale, target_scale, rtol=1e-7)
