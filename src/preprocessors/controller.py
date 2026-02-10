"""Preprocessor Controllers.

This module contains controller functions that orchestrate scan processing pipelines.

.. seealso::

    :func:`process_scan_controller`
        Process scan files to generate surface maps and previews.
    :func:`process_prepare_mark`
        Prepare striation mark data for analysis.
"""

from collections.abc import Iterable
from pathlib import Path

from container_models.base import UnitVector
from container_models.image import ImageContainer
from loguru import logger
from mutations.shading import ImageForDisplay, LightIntensityMap
from returns.pipeline import pipe

from extractors.schemas import ProcessedDataAccess
from pipelines import run_pipeline


def _run_preview(image: ImageContainer, output_path: Path) -> None:
    run_pipeline(
        pipe(ImageForDisplay(std_scaler=2.0)),
        image,
        "Failed to extract image for preview",
    ).export_png(output_path)


def process_scan_controller(
    scan_file: Path, output_path: Path, light_sources: Iterable[UnitVector], observer: UnitVector
) -> None:
    """Process a scan file to generate a surface map and preview image, saving them to the specified output path."""
    files = ProcessedDataAccess.get_files(output_path)

    image = ImageContainer.from_scan_file(scan_file)
    image.export_x3p(files["scan"])

    run_pipeline(
        pipe(LightIntensityMap(light_sources, observer)),
        image,
        "Failed to create the surface map",
    ).export_png(files["surface_map"])
    _run_preview(image, files["preview"])


def process_prepare_mark(scan_file: Path, files: dict[str, Path]) -> dict[str, Path]:
    """Prepare striation mark data."""
    parsed_scan = ImageContainer.from_scan_file(scan_file)
    # rotate and crop function()
    # resample()
    logger.info("Preparing mark")
    # surface_map_pipeline(
    #     parsed_scan=parsed_scan,
    #     output_path=files["surface_map"],
    #     # TODO: make parameters needed explicit so we supply needed arguments.
    # )
    _run_preview(parsed_scan, files["preview"])
    return files
