from collections.abc import Callable
from pathlib import Path

import numpy as np
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType
from conversion.leveling.solver.utils import compute_image_center
from conversion.resample import resample_mark
from conversion.rotate import rotate_crop_and_mask_image_by_crop
from loguru import logger
from mutations import CropToMask, GausianRegressionFilter, LevelMap, Mask, Resample
from skimage.transform import resize

from constants import LIGHT_SOURCES, OBSERVER
from preprocessors.pipelines import parse_scan_pipeline, preview_pipeline, surface_map_pipeline, x3p_pipeline
from preprocessors.schemas import (
    EditImage,
    PrepareMarkImpression,
    PrepareMarkStriation,
)


def process_prepare_mark(
    scan_file: Path,
    marking_method: Callable[[Mark], tuple[Mark, Mark]],
    params: PrepareMarkImpression | PrepareMarkStriation,
    files: dict[str, Path],
) -> dict[str, Path]:
    """Prepare striation and impression mark data."""
    parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
    mask = np.ones(parsed_scan.data.shape)  # fix
    rotated_image = rotate_crop_and_mask_image_by_crop(
        scan_image=parsed_scan, mask=mask, bounding_box=params.bounding_box
    )
    mark = Mark(
        scan_image=rotated_image,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,  # fix
    )
    mark = resample_mark(mark)
    logger.info("Preparing mark")
    processed_mark, _ = marking_method(mark)
    logger.info("saving x3p, surface_map.png and preview.png")
    surface_map_pipeline(
        parsed_scan=processed_mark.scan_image,
        output_path=files["surface_map"],
        observer=OBSERVER,
        light_sources=LIGHT_SOURCES,
    )
    preview_pipeline(parsed_scan=processed_mark.scan_image, output_path=files["preview"])
    x3p_pipeline(parsed_scan=processed_mark.scan_image, output_path=files["scan"])
    return files


def edit_scan_image(scan_image: ScanImage, edit_image_params: EditImage):
    """From a scan_image file to an edited image file."""
    output_shape = (
        1 / edit_image_params.resampling_factor * scan_image.height,
        1 / edit_image_params.resampling_factor * scan_image.width,
    )
    resampled_mask = np.asarray(
        resize(
            image=edit_image_params.mask_array,
            output_shape=output_shape,
            mode="edge",
            anti_aliasing=False,
        ),
        dtype=np.bool_,
    )
    reference_point_x, reference_point_y = compute_image_center(scan_image)
    pipeline = [
        Resample(target_shape=output_shape),
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
