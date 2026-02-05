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

from preprocessors.pipelines import parse_scan_pipeline, preview_pipeline
from preprocessors.schemas import Mask as MaskType


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
    scan_file: Path,
    terms: SurfaceTerms,
    crop: bool,
    mask: MaskType,
    resampling_factor: float,
    step_size_x: int,
    step_size_y: int,
):
    """From a scan_image file to a editted image file."""
    scan_image = parse_scan_pipeline(scan_file, step_size_x, step_size_y)

    resample_mutator = Resample(x_factor=resampling_factor, y_factor=resampling_factor)
    # TODO: the quick fix, what needs to be solved by ImageContainer instead of ScanImage when branch of sharlong is merged
    mask_image = ScanImage(data=np.array(mask), scale_x=scan_image.scale_x, scale_y=scan_image.scale_y)
    resampled_mask = cast(BinaryMask, resample_mutator(scan_image=mask_image).unwrap().data)

    pipeline = pipe([
        resample_mutator,
        CropToMask(mask=resampled_mask) if crop else Mask(mask=resampled_mask),
        LevelMap(x_reference_point=1, y_reference_point=1, terms=terms),
    ])  # TODO: refer to our own pipeline maker when <insert branch of sharlon>
    return pipeline(scan_image)
