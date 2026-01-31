from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, PositiveInt
from returns.io import impure_safe
from returns.result import safe
from scipy.constants import micro
from skimage.transform import resize
from surfalize import Surface
from surfalize.file import FileHandler
from surfalize.file.al3d import MAGIC

from parsers.patches.al3d import read_al3d
from image_tasks.factory import create_image_task
from image_tasks.types.base import DepthData, FloatArray2D
from image_tasks.types.scan_image import MetaData, ScanImage
from utils.logger import log_railway_function

# register the patched method as a parser
FileHandler.register_reader(suffix=".al3d", magic=MAGIC)(read_al3d)


class SubSampleParams(BaseModel):
    step_size_x: PositiveInt
    step_size_y: PositiveInt


@lru_cache(maxsize=1)
@log_railway_function(
    "Failed to load image file",
    "Successfully loaded scan file",
)
@impure_safe
def load_scan_image(scan_file: Path) -> ScanImage:
    """
    Load a scan image from a file. Parsed values will be converted to meters (m).

    :param scan_file: The path to the file containing the scanned image data.
    :returns: An instance of `ScanImage`.
    """
    surface = Surface.load(scan_file)

    return ScanImage(
        data=np.asarray(surface.data, dtype=np.float64) * micro,
        meta_data=MetaData.model_validate(
            {"scale": {"x": surface.step_x * micro, "y": surface.step_y * micro}}
            | surface.metadata
        ),
    )


def _upsample_image_data(data: DepthData, shape: tuple[int, int]) -> FloatArray2D:
    """Upsample image data in a `ScanImage` instance to a common target scale."""
    return np.asarray(
        resize(
            image=data,
            output_shape=shape,
            mode="edge",
            anti_aliasing=False,  # Disabled for pure upsampling
            preserve_range=True,  # Keep original data intensity levels
            order=0,  # Nearest Neighbor so that NaNs appear at corresponding coordinates
        ),
        dtype=np.float64,
    )


@safe
def _make_isotropic(scan_image: ScanImage) -> ScanImage:
    """
    Resample a scan image to isotropic resolution (i.e. equal pixel spacing in X and Y).

    If the scan image is already isotropic, the original instance is returned.
    Otherwise, the image data is upsampled to the highest available resolution
    (the smaller of the two scale factors) using nearest-neighbor interpolation.
    Note: NaN values are preserved and will not be interpolated.

    :param scan_image: The ScanImage instance to be resampled.
    :returns: A new ScanImage instance with isotropic scaling.
    """
    target_scale = scan_image.target_scale("upscale")
    scan_image.data = _upsample_image_data(
        scan_image.data, scan_image.rescale_shape(target_scale)
    )
    scan_image.meta_data.scale.x = target_scale
    scan_image.meta_data.scale.y = target_scale
    return scan_image


make_isotropic = create_image_task(
    _make_isotropic,
    skip_predicate=lambda scan_image, **_: scan_image.meta_data.is_isotropic,
    failure_msg="Failed to make image isotropic",
    success_msg="Successfully made image isotropic",
)


@safe
def _subsample_scan_image(
    scan_image: ScanImage, *, step_size_x: int, step_size_y: int
) -> ScanImage:
    """
    Subsample the data in a `ScanImage` instance by skipping steps in each dimension.

    :param scan_image: The instance of `ScanImage` containing the 2D image data to subsample.
    :param step_size_x: The number of steps to skip in the X-direction.
    :param step_size_y: The number of steps to skip in the Y-direction.
    :returns: A subsampled `ScanImage` with updated scales. If both step sizes are 1, the original
        `ScanImage` instance is returned.
    """
    width, height = scan_image.width, scan_image.height
    if not (0 < step_size_x < width and 0 < step_size_y < height):
        raise ValueError(
            f"Step size should be positive and smaller than the image size: {(height, width)}"
        )
    logger.info(
        "Subsampling scan image with step sizes x: {step_size_x}, y: {step_size_y}"
    )
    scan_image.data = scan_image.data[::step_size_y, ::step_size_x].copy()
    scan_image.meta_data.scale.x *= step_size_x
    scan_image.meta_data.scale.y *= step_size_y
    return scan_image


subsample_scan_image = create_image_task(
    _subsample_scan_image,
    params_model=SubSampleParams,
    skip_predicate=lambda step_size_x, step_size_y, **_: (
        step_size_x == 1 and step_size_y == 1
    ),
    failure_msg="Failed to subsample image",
    success_msg="Successfully subsampled image",
)
