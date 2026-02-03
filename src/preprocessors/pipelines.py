from collections.abc import Iterable
from pathlib import Path

from container_models.image import ImageContainer
from container_models.light_source import LightSource
from mutations.sampling import IsotropicResample, Subsample
from mutations.shading import GrayScale, ImageForDisplay, LightIntensityMap
from parsers import load_scan_image, parse_to_x3p
from renders.image_io import save_image, save_x3p
from returns.curry import partial

from pipelines import run_pipeline
from preprocessors.schemas import (
    PreprocessingImpressionParams,
    PreprocessingStriationParams,
)


def parse_scan_pipeline(scan_file: Path, step_size_x: int, step_size_y: int) -> ImageContainer:
    """
    Parse a scan file and load it as a ImageContainer.

    :param scan_file: The path to the scan file to parse.
    :param parameters: All parameters used in the pipeline.
    :return: The parsed scan image data.
    :raises HTTPException: If the file cannot be parsed or read.
    """
    return run_pipeline(
        scan_file,
        load_scan_image,
        Subsample(step_size_x=step_size_x, step_size_y=step_size_y),
        IsotropicResample,
        error_message=f"Failed to parsed given scan file: {scan_file}",
    )


def x3p_pipeline(parsed_scan: ImageContainer, output_path: Path) -> Path:
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


def surface_map_pipeline(  # noqa
    parsed_scan: ImageContainer,
    output_path: Path,
    light_sources: Iterable[LightSource],
    observer: LightSource,
) -> Path:
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
        LightIntensityMap(
            (light.unit_vector for light in light_sources),
            observer.unit_vector,
        ),
        GrayScale,
        partial(save_image, output_path=output_path),
        error_message=f"Failed to create the surface map: {output_path}",
    )


def preview_pipeline(parsed_scan: ImageContainer, output_path: Path) -> Path:
    """
    Generate a preview image from scan data and save it to the specified path.

    :param parsed_scan: The scan image data to generate a preview from.
    :param output_path: The file path where the preview image will be saved.
    :return: The path to the saved preview image file.
    :raises HTTPException: If image generation or saving fails.
    """
    return run_pipeline(
        parsed_scan,
        ImageForDisplay(std_scaler=2.0),
        GrayScale,
        partial(save_image, output_path=output_path),
        error_message=f"Failed to create the surface map: {output_path}",
    )


def impression_mark_pipeline(params: PreprocessingImpressionParams) -> Path:
    """PLACEHOLDER."""  # noqa: D401
    return Path()  # TODO: fill in when implementing impression mark.


def striation_mark_pipeline(params: PreprocessingStriationParams) -> Path:
    """PLACEHOLDER."""  # noqa: D401
    return Path()  # TODO: fill in when implementing striation mark.
