from pathlib import Path

import numpy as np
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox, Mark, MarkType
from conversion.export.mark import save_mark
from conversion.export.profile import save_profile
from conversion.leveling.solver.utils import compute_image_center
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from conversion.preprocess_impression.preprocess_impression import preprocess_impression_mark
from conversion.preprocess_striation import PreprocessingStriationParams
from conversion.preprocess_striation.pipeline import preprocess_striation_mark
from conversion.resample import resample_mark
from conversion.rotate import rotate_crop_and_mask_image_by_crop
from loguru import logger
from mutations import CropToMask, GausianRegressionFilter, LevelMap, Mask, Resample
from skimage.transform import resize

from constants import LIGHT_SOURCES, OBSERVER
from preprocessors.pipelines import parse_scan_pipeline, preview_pipeline, surface_map_pipeline
from preprocessors.schemas import EditImage


def _extract_mark_from_scan(scan_file: Path, mark_type: MarkType, mask: BinaryMask, bounding_box: BoundingBox) -> Mark:
    """Parse a scan file and extract a mark by rotating, cropping, masking, and resampling."""
    logger.info("Parsing scan image")
    parsed_scan = parse_scan_pipeline(scan_file, 1, 1)
    logger.info("Rotating and cropping scan image")
    rotated_image = rotate_crop_and_mask_image_by_crop(scan_image=parsed_scan, mask=mask, bounding_box=bounding_box)
    logger.info("Transforming scan image to mark")
    mark = Mark(
        scan_image=rotated_image,
        mark_type=mark_type,
    )
    mark = resample_mark(mark)
    return mark


def _save_outputs(mark: Mark, processed_mark: Mark, files: dict[str, Path]) -> None:
    """Save surface map, preview, raw mark, and processed mark."""
    logger.info("Saving marks, surface_map.png and preview.png")
    surface_map_pipeline(
        parsed_scan=processed_mark.scan_image,
        output_path=files["surface_map"],
        observer=OBSERVER,
        light_sources=LIGHT_SOURCES,
    )
    preview_pipeline(parsed_scan=processed_mark.scan_image, output_path=files["preview"])
    save_mark(mark, path=files["mark_data"])
    save_mark(processed_mark, path=files["processed_data"])


def process_prepare_impression_mark(  # noqa: PLR0913
    scan_file: Path,
    mark_type: MarkType,
    mask: BinaryMask,
    bounding_box: BoundingBox,
    preprocess_parameters: PreprocessingImpressionParams,
    files: dict[str, Path],
) -> dict[str, Path]:
    """Prepare impression mark data."""
    mark = _extract_mark_from_scan(scan_file, mark_type, mask, bounding_box)
    logger.info("Preparing mark")
    processed_mark, leveled_mark = preprocess_impression_mark(mark, params=preprocess_parameters)
    _save_outputs(mark, processed_mark, files)
    save_mark(leveled_mark, path=files["leveled_data"])
    return files


def process_prepare_striation_mark(  # noqa: PLR0913
    scan_file: Path,
    mark_type: MarkType,
    mask: BinaryMask,
    bounding_box: BoundingBox,
    preprocess_parameters: PreprocessingStriationParams,
    files: dict[str, Path],
) -> dict[str, Path]:
    """Prepare striation mark data."""
    mark = _extract_mark_from_scan(scan_file, mark_type, mask, bounding_box)
    logger.info("Preparing mark")
    processed_mark, profile = preprocess_striation_mark(mark, params=preprocess_parameters)
    _save_outputs(mark, processed_mark, files)
    save_profile(profile, path=files["profile_data"])
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
