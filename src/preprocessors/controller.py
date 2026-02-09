from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms
from loguru import logger
from mutations import CropToMask, LevelMap, Mask, Resample
from returns.pipeline import pipe
from skimage.transform import resize

from preprocessors.pipelines import parse_scan_pipeline, preview_pipeline


def process_prepare_mark(
    scan_file: Path, marking_method: Callable[..., Path], files: dict[str, Path]
) -> dict[str, Path]:
    """Prepare striation mark data."""
    parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
    # rotate and crop function()
    # resample()
    logger.info("Preparing mark")
    marking_method()
    # surface_map_pipeline(
    #     parsed_scan=parsed_scan,
    #     output_path=files["surface_map"],
    #     # TODO: make parameters needed explicit so we supply needed arguments.
    # )
    preview_pipeline(parsed_scan=parsed_scan, output_path=files["preview"])
    return files


def edit_image_pipeline(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    crop: bool,
    mask: BinaryMask,
    resampling_factor: float,
):
    """From a scan_image file to an edited image file."""
    resampled_mask = cast(
        BinaryMask,
        resize(
            image=mask,
            output_shape=(1 / resampling_factor * scan_image.height, 1 / resampling_factor * scan_image.width),
            mode="edge",
            anti_aliasing=False,
        ),
    ).astype(np.bool_)
    pipeline = pipe([
        Resample(x_factor=resampling_factor, y_factor=resampling_factor),
        Mask(mask=resampled_mask),
        *([CropToMask(mask=resampled_mask)] if crop else []),
        LevelMap(x_reference_point=1, y_reference_point=1, terms=terms),
    ])
    return pipeline(scan_image)
