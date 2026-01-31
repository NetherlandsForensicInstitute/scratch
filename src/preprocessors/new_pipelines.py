from pathlib import Path

from image_tasks.loaders import load_scan_image, make_isotropic, subsample_scan_image
from image_tasks.types.scan_image import ScanImage

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
        make_isotropic(),
        error_message=f"Failed to parsed given scan file: {scan_file}",
    )
