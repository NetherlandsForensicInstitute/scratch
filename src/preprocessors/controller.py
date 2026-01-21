from collections.abc import Callable
from pathlib import Path

from loguru import logger

from preprocessors.pipelines import parse_scan_pipeline, preview_pipeline, surface_map_pipeline, x3p_pipeline
from preprocessors.schemas import UploadScanParameters


def process_prepare_mark(
    scan_file: Path, marking_method: Callable[..., Path], files: dict[str, Path]
) -> dict[str, Path]:
    """Prepare striation mark data."""
    parsed_scan = parse_scan_pipeline(
        scan_file, UploadScanParameters.model_construct()
    )  # TODO: add / modify pipline for no subsampling and saving.
    x3p_pipeline(parsed_scan, files["scan"])
    # rotate and crop function()
    # resample()
    logger.info("Preparing mark")
    marking_method()
    surface_map_pipeline(
        parsed_scan=parsed_scan,
        output_path=files["surface_map"],
        parameters=UploadScanParameters.model_construct(),
        # TODO: make parameters needed explicit so we supply needed arguments.
    )
    preview_pipeline(parsed_scan=parsed_scan, output_path=files["preview"])
    return files
