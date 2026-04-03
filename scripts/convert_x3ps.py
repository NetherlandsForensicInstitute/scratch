"""
Convert measurement.x3p files from MATLAB result folders via the preprocessor API.

Walks a nested folder structure, converts x3p files, and calls the
preprocessor API to generate preview and surface map images.
"""

import argparse
import contextlib
import logging
import os
import uuid
from pathlib import Path

import requests
from container_models.scan_image import ScanImage
from parsers import convert_to_x3p

from scripts.conversion_utils import (
    ConversionConfig,
    copy_db_scratch_files,
    find_mark_folders,
    flatten_processed_folders,
    load_shape,
    run_parallel,
    save_shape,
)
from scripts.http_utils import _cleanup_vault, _post_with_retry

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def convert_measurement_x3p(measurement_folder: Path, cfg: ConversionConfig) -> tuple[Path, tuple[int, int]]:
    """Convert a measurement.x3p and generate preview/surface_map images.

    :returns: (path to converted x3p, (size_x, size_y) pixel dimensions).
    """
    original = measurement_folder / "measurement.x3p"
    output_x3p = cfg.output_dir / measurement_folder.relative_to(cfg.root) / "measurement.x3p"

    if output_x3p.exists() and not cfg.force:
        shape = load_shape(output_x3p)
        if shape is not None:
            return output_x3p, shape
        logger.warning("Missing shape file for %s, reconverting", output_x3p)

    output_x3p.parent.mkdir(parents=True, exist_ok=True)

    shape = convert_x3p(original, output_x3p)
    save_shape(output_x3p, shape)

    result = _post_with_retry(f"{cfg.api_url}/preprocessor/process-scan", {"scan_file": str(output_x3p)})
    for key in ("preview_image", "surface_map_image"):
        if key in result:
            url = result[key]
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            (output_x3p.parent / url.rsplit("/", 1)[-1]).write_bytes(resp.content)
    _cleanup_vault(result)
    return output_x3p, shape


def main() -> None:
    """Entry point: convert all measurement.x3p files found under root."""
    parser = argparse.ArgumentParser(description="Convert measurement.x3p files via Python API")
    parser.add_argument("root", type=Path, help="Root folder to search")
    parser.add_argument("output", type=Path, help="Output folder (mirrors input structure)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Preprocessor API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Reconvert even if output exists")
    args = parser.parse_args()

    cfg = ConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    copy_db_scratch_files(cfg.root, cfg.output_dir)

    marks = list(find_mark_folders(cfg.root))
    unique_measurements = list({mf for mf, _ in marks})
    logger.info(f"Found {len(unique_measurements)} unique measurements")

    run_parallel(
        ((mf, convert_measurement_x3p, (mf, cfg)) for mf in unique_measurements),
        args.workers,
        "Converting x3p",
        " files",
    )

    flatten_processed_folders(cfg.output_dir)
    logger.info(f"Done: converted {len(unique_measurements)} x3p files")


if __name__ == "__main__":
    main()


def convert_x3p(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Load an X3P from path, parse it, and write the result atomically."""
    scan = ScanImage.from_file(input_path)
    x3p = convert_to_x3p(scan)

    tmp = output_path.with_stem(f".{uuid.uuid4().hex}.tmp")
    try:
        x3p.write(str(tmp))
        os.replace(tmp, output_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise

    return scan.width, scan.height
