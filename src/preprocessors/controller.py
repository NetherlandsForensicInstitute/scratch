from pathlib import Path

import numpy as np
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox, Mark, MarkImpression, MarkStriation
from conversion.export.mark import save_mark
from conversion.export.profile import save_profile
from conversion.leveling.solver.utils import compute_image_center
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from conversion.preprocess_impression.preprocess_impression import preprocess_impression_mark
from conversion.preprocess_striation import PreprocessingStriationParams
from conversion.preprocess_striation.pipeline import preprocess_striation_mark
from conversion.resample import resample_mark
from loguru import logger
from mutations import CropToMask, GaussianRegressionFilter, LevelMap, Mask, Resample, Rotate
from mutations.base import ImageMutation
from mutations.filter import FilterNeedles
from scipy.constants import micro
from skimage.transform import resize

from constants import LIGHT_SOURCES, OBSERVER
from preprocessors.constants import PrepareMarkImpressionFiles, PrepareMarkStriationFiles
from preprocessors.pipelines import preview_pipeline, surface_map_pipeline
from preprocessors.schemas import EditImage


def _scan_image_to_mark(mask: BinaryMask, bounding_box: BoundingBox | None, scan_image: ScanImage) -> ScanImage:
    if bounding_box is None:
        logger.info("No bounding_box given. skipping rotating image.")
        masking = Mask(mask=mask)
        filtering = FilterNeedles()
        cropping = CropToMask(mask=mask)
        pipeline: list[ImageMutation] = [masking, filtering, cropping]
    else:
        masking = Mask(mask=mask)
        filtering = FilterNeedles()
        rotating = Rotate.from_bounding_box(bounding_box=bounding_box)
        cropping = CropToMask.from_rotation(mask_before_rotation=mask, rotation_angle=rotating.rotation_angle)
        pipeline: list[ImageMutation] = [masking, filtering, rotating, cropping]

    for image_mutation in pipeline:
        logger.info(f"mutating scan_image with : {image_mutation.__class__.__name__}")
        scan_image = image_mutation(scan_image)
    return scan_image


def _extract_mark_from_scan(
    scan_image: ScanImage, mark_type: MarkImpression | MarkStriation, mask: BinaryMask, bounding_box: BoundingBox | None
) -> Mark:
    """Parse a scan file and extract a mark by rotating, cropping, masking, and resampling."""
    logger.debug("mutating scan_image with given parameters")
    scan_image = _scan_image_to_mark(mask=mask, bounding_box=bounding_box, scan_image=scan_image)
    logger.info("Transforming scan image to mark")
    mark = Mark(
        scan_image=scan_image,
        mark_type=mark_type,
    )
    mark = resample_mark(mark)
    return mark


def _save_outputs(
    mark: Mark,
    processed_mark: Mark,
    working_dir: Path,
    files: type[PrepareMarkStriationFiles | PrepareMarkImpressionFiles],
) -> None:
    """Save surface map, preview, raw mark, and processed mark."""
    logger.info("Saving marks, surface_map.png and preview.png")
    surface_map_pipeline(
        parsed_scan=processed_mark.scan_image,
        output_path=files.surface_map_image.get_file_path(working_dir),
        observer=OBSERVER,
        light_sources=LIGHT_SOURCES,
    )
    preview_pipeline(
        parsed_scan=processed_mark.scan_image,
        output_path=files.preview_image.get_file_path(working_dir),
    )
    save_mark(mark, path=files.mark_data.get_file_path(working_dir))
    save_mark(processed_mark, path=files.processed_data.get_file_path(working_dir))


def process_prepare_impression_mark(  # noqa: PLR0913
    scan_image: ScanImage,
    mark_type: MarkImpression,
    mask: BinaryMask,
    bounding_box: BoundingBox | None,
    preprocess_parameters: PreprocessingImpressionParams,
    working_dir: Path,
) -> None:
    """Prepare impression mark data."""
    mark = _extract_mark_from_scan(scan_image, mark_type, mask, bounding_box)
    logger.info("Preparing mark")
    processed_mark, leveled_mark = preprocess_impression_mark(mark, params=preprocess_parameters)
    _save_outputs(mark, processed_mark, working_dir, files=PrepareMarkImpressionFiles)
    save_mark(leveled_mark, path=PrepareMarkImpressionFiles.leveled_data.get_file_path(working_dir))


def process_prepare_striation_mark(  # noqa: PLR0913
    scan_image: ScanImage,
    mark_type: MarkStriation,
    mask: BinaryMask,
    bounding_box: BoundingBox | None,
    preprocess_parameters: PreprocessingStriationParams,
    working_dir: Path,
) -> None:
    """Prepare striation mark data."""
    mark = _extract_mark_from_scan(scan_image, mark_type, mask, bounding_box)
    logger.info("Preparing mark")
    processed_mark, profile = preprocess_striation_mark(mark, params=preprocess_parameters)
    _save_outputs(mark, processed_mark, working_dir, files=PrepareMarkStriationFiles)
    save_profile(profile, path=PrepareMarkStriationFiles.profile_data.get_file_path(working_dir))


def edit_scan_image(scan_image: ScanImage, edit_image_params: EditImage, mask: BinaryMask) -> ScanImage:
    """From a scan_image file to an edited image file."""
    output_shape = (
        1 / edit_image_params.resampling_factor * scan_image.height,
        1 / edit_image_params.resampling_factor * scan_image.width,
    )
    resampled_mask = np.asarray(
        resize(
            image=mask,
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
            x_reference_point=reference_point_x,
            y_reference_point=reference_point_y,
            terms=edit_image_params.terms.to_surface_terms(),
        ),
        GaussianRegressionFilter(
            regression_order=edit_image_params.regression_order,
            cutoff_length=edit_image_params.cutoff_length * micro,
            is_high_pass=True,
        ),
    ]
    logger.debug(f"mutations to be applied on the scan image:{[item.__class__.__name__ for item in pipeline]}")
    for mutation in pipeline:
        logger.debug(f"Mutating the image with: {mutation.__class__.__name__}")
        scan_image = mutation(scan_image)
    return scan_image
