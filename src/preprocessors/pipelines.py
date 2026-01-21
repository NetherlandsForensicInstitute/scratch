from functools import partial
from pathlib import Path

from container_models.scan_image import ScanImage
from parsers import load_scan_image, parse_to_x3p, save_x3p, subsample_scan_image
from renders import (
    apply_multiple_lights,
    compute_surface_normals,
    get_scan_image_for_display,
    save_image,
    scan_to_image,
)
from renders.normalizations import normalize_2d_array

from pipelines import run_pipeline
from preprocessors.schemas import PreprocessingImpressionParams, PreprocessingStriationParams, UploadScanParameters


def parse_scan_pipeline(scan_file: Path, parameters: UploadScanParameters) -> ScanImage:
    """
    Parse a scan file and load it as a ScanImage.

    :param scan_file: The path to the scan file to parse.
    :param parameters: All parameters used in the pipeline.
    :return: The parsed scan image data.
    :raises HTTPException: If the file cannot be parsed or read.
    """
    return run_pipeline(
        scan_file,
        load_scan_image,
        partial(subsample_scan_image, **parameters.as_dict(include={"step_size_x", "step_size_y"})),
        error_message=f"Failed to parsed given scan file: {scan_file}",
    )


def x3p_pipeline(parsed_scan: ScanImage, output_path: Path) -> Path:
    """
    Convert a scan image to X3P format and save it to the specified path.

    :param parsed_scan: The scan image data to convert to X3P format.
    :param output_path: The file path where the X3P file will be saved.
    :return: The path to the saved X3P file.
    :raises HTTPException: If conversion or saving fails.
    """
    return run_pipeline(
        parsed_scan,
        parse_to_x3p,
        partial(save_x3p, output_path=output_path),
        error_message=f"Failed to create the x3p: {output_path}",
    )


def surface_map_pipeline(parsed_scan: ScanImage, output_path: Path, parameters: UploadScanParameters) -> Path:
    """
    Generate a 3D surface map image from scan data and save it to the specified path.

    :param parsed_scan: The scan image data to generate a surface map from.
    :param output_path: The file path where the surface map image will be saved.
    :param parameters: All parameters used in the pipeline.
    :return: The path to the saved surface map image file.
    :raises HTTPException: If image generation or saving fails.
    """
    return run_pipeline(
        parsed_scan,
        compute_surface_normals,
        partial(apply_multiple_lights, **parameters.as_dict(exclude={"step_size_x", "step_size_y"})),
        normalize_2d_array,
        scan_to_image,
        partial(save_image, output_path=output_path),
        error_message=f"Failed to create the surface map: {output_path}",
    )


def preview_pipeline(parsed_scan: ScanImage, output_path: Path) -> Path:
    """
    Generate a preview image from scan data and save it to the specified path.

    :param parsed_scan: The scan image data to generate a preview from.
    :param output_path: The file path where the preview image will be saved.
    :return: The path to the saved preview image file.
    :raises HTTPException: If image generation or saving fails.
    """
    return run_pipeline(
        parsed_scan,
        get_scan_image_for_display,
        scan_to_image,
        partial(save_image, output_path=output_path),
        error_message=f"Failed to create the surface map: {output_path}",
    )


def impression_mark_pipeline(params: PreprocessingImpressionParams) -> Path:
    """PLACEHOLDER."""  # noqa: D401
    return Path()  # TODO: fill in when implementing impression mark.


def striation_mark_pipeline(params: PreprocessingStriationParams) -> Path:
    """PLACEHOLDER."""  # noqa: D401
    return Path()  # TODO: fill in when implementing striation mark.
