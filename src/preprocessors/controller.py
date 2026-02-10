from collections.abc import Callable
from pathlib import Path

import numpy as np
from container_models.scan_image import ScanImage
from conversion.leveling.solver.utils import compute_image_center
from loguru import logger
from mutations import CropToMask, GausianRegressionFilter, LevelMap, Mask, Resample
from skimage.transform import resize

from preprocessors.pipelines import parse_scan_pipeline, preview_pipeline
from preprocessors.schemas import EditImage


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


def edit_scan_image(scan_image: ScanImage, edit_image_params: EditImage):
    """From a scan_image file to an edited image file."""
    resampled_mask = np.asarray(
        resize(
            image=edit_image_params.mask_array,
            output_shape=(
                1 / edit_image_params.resampling_factor * scan_image.height,
                1 / edit_image_params.resampling_factor * scan_image.width,
            ),
            mode="edge",
            anti_aliasing=False,
        ),
        dtype=np.bool_,
    )
    reference_point_x, reference_point_y = compute_image_center(scan_image)
    pipeline = [
        Resample(x_factor=edit_image_params.resampling_factor, y_factor=edit_image_params.resampling_factor),
        Mask(mask=resampled_mask),
        *([CropToMask(mask=resampled_mask)] if edit_image_params.crop else []),
        LevelMap(
            x_reference_point=reference_point_x, y_reference_point=reference_point_y, terms=edit_image_params.terms
        ),
        GausianRegressionFilter(
            regression_order=edit_image_params.regression_order, cutoff_length=edit_image_params.cutoff_length
        ),
    ]
    logger.debug(f"mutations to be applied on the scan image:{[item.__class__.__name__ for item in pipeline]}")
    for mutation in pipeline:
        logger.debug(f"Mutating the image with: {mutation.__class__.__name__}")
        scan_image = mutation(scan_image).unwrap()
    return scan_image
