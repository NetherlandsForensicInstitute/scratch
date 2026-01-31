from pathlib import Path

from container_models.scan_image import ScanImage
from tasks.loaders import Isotropic, SubSample, load_scan_image

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
        SubSample(step_size_x, step_size_y),
        Isotropic(),
        error_message=f"Failed to parsed given scan file: {scan_file}",
    )
