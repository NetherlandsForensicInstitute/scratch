from collections.abc import Iterable
from pathlib import Path

from container_models.light_source import LightSource
from image_tasks.loaders import load_scan_image, make_isotropic, subsample_scan_image
from image_tasks.renders import apply_multiple_lights
from image_tasks.types.scan_image import Point, ScanImage
from renders import save_image, scan_to_image
from returns.curry import partial

from pipelines import run_pipeline


def parse_scan_pipeline(scan_file: Path, step_size_x: int, step_size_y: int) -> ScanImage:
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
        subsample_scan_image(step_size_x=step_size_x, step_size_y=step_size_y),
        make_isotropic,
        error_message=f"Failed to parsed given scan file: {scan_file}",
    )


def surface_map_pipeline(  # noqa
    parsed_scan: ScanImage,
    output_path: Path,
    light_sources: Iterable[LightSource],
    observer: LightSource,
    scale: Point[float],
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
        apply_multiple_lights(
            light_sources=light_sources,
            observer=observer,
            scale=scale,
        ),
        scan_to_image,
        partial(save_image, output_path=output_path),
        error_message=f"Failed to create the surface map: {output_path}",
    )
