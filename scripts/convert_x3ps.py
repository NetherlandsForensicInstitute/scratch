"""
Convert measurement.x3p files from MATLAB result folders via the preprocessor API.

Walks a nested folder structure, converts x3p files, and calls the
preprocessor API to generate preview and surface map images.

Also supports a --skip-conversion mode: if you already have converted
measurement.x3p files under root (e.g. only outputs remain, originals
are gone), this finds them, copies each to the mirrored path under
output, and just (re)runs them through the preprocessor API — without
trying to parse/re-encode via the normal conversion step.
"""

import argparse
import contextlib
import logging
import os
import shutil
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
from scripts.http_utils import _post_with_retry, download_result_files

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PREVIEW_KEYS = ("preview_image", "surface_map_image")


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


def convert_measurement_x3p(
    measurement_folder: Path,
    cfg: ConversionConfig,
    session: requests.Session,
    *,
    skip_conversion: bool = False,
) -> tuple[Path, tuple[int, int] | None]:
    """Convert a measurement.x3p (or copy an existing one) and generate preview/surface_map images.

    :param skip_conversion: if True, ``measurement_folder`` is assumed to already contain a
        converted ``measurement.x3p`` (e.g. the "original" no longer exists, just the x3p does).
        Instead of parsing/re-encoding via ``convert_x3p``, the existing file is plainly copied
        from root to the mirrored path under output, then sent to the preprocessor API.
    :returns: (path to the x3p under output, (size_x, size_y) pixel dimensions, or None if unknown).
    """
    original = measurement_folder / "measurement.x3p"
    output_x3p = cfg.output_dir / measurement_folder.relative_to(cfg.root) / "measurement.x3p"

    if skip_conversion:
        if not original.exists():
            raise FileNotFoundError(f"Expected an existing x3p at {original}, but it's missing")

        if output_x3p.exists() and not cfg.force:
            shape = load_shape(output_x3p)
        else:
            output_x3p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, output_x3p)
            shape = load_shape(output_x3p)  # likely None — no fresh conversion happened to produce one
    else:
        if output_x3p.exists() and not cfg.force:
            shape = load_shape(output_x3p)
            if shape is not None:
                return output_x3p, shape
            logger.warning("Missing shape file for %s, reconverting", output_x3p)

        output_x3p.parent.mkdir(parents=True, exist_ok=True)
        shape = convert_x3p(original, output_x3p)
        save_shape(output_x3p, shape)

    result = _post_with_retry(f"{cfg.api_url}/preprocessor/process-scan", {"scan_file": str(output_x3p)})
    downloaded = download_result_files(result, session, keys=_PREVIEW_KEYS)
    for filename, content in downloaded.items():
        (output_x3p.parent / filename).write_bytes(content)

    return output_x3p, shape


def _convert_measurement_safe(
    measurement_folder: Path,
    cfg: ConversionConfig,
    session: requests.Session,
    skip_conversion: bool = False,
) -> tuple[Path, tuple[int, int] | None] | None:
    """Convert measurement while logging errors instead of raising."""
    try:
        return convert_measurement_x3p(measurement_folder, cfg, session, skip_conversion=skip_conversion)
    except Exception:
        logger.exception("Failed to process %s", measurement_folder)
        return None


def find_existing_measurement_folders(root_dir: Path) -> list[Path]:
    """Find folders directly containing an already-converted measurement.x3p.

    Used in --skip-conversion mode to walk the *input* root dir (since there's no
    "original-to-convert" structure there to walk via find_mark_folders, just whatever
    already-converted x3p files are sitting on disk) and copy each over to output.
    """
    return sorted({p.parent for p in root_dir.rglob("measurement.x3p")})


def main() -> None:
    """Entry point: convert (or reprocess) all measurement.x3p files found under root."""
    parser = argparse.ArgumentParser(description="Convert measurement.x3p files via Python API")
    parser.add_argument("root", type=Path, help="Root folder to search")
    parser.add_argument("output", type=Path, help="Output folder (mirrors input structure)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Preprocessor API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Reconvert even if output exists")
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help=(
            "Don't try to convert from an 'original' file. Instead, treat the existing "
            "measurement.x3p files under root as already-converted: find them, copy them "
            "to the mirrored path under output, and just (re)run them through the "
            "preprocessor API."
        ),
    )
    args = parser.parse_args()

    cfg = ConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    copy_db_scratch_files(cfg.root, cfg.output_dir)

    if args.skip_conversion:
        unique_measurements = find_existing_measurement_folders(cfg.root)
        logger.info(f"Found {len(unique_measurements)} existing measurement.x3p files")
    else:
        marks = list(find_mark_folders(cfg.root))
        unique_measurements = list({mf for mf, _ in marks})
        logger.info(f"Found {len(unique_measurements)} unique measurements")

    session = requests.Session()
    try:
        results = run_parallel(
            ((mf, _convert_measurement_safe, (mf, cfg, session, args.skip_conversion)) for mf in unique_measurements),
            args.workers,
            "Converting x3p" if not args.skip_conversion else "Reprocessing x3p",
            " files",
        )
    finally:
        session.close()

    failed = sum(1 for v in results.values() if v is None)
    succeeded = len(results) - failed
    if failed:
        logger.warning("%d/%d measurements failed", failed, len(results))

    flatten_processed_folders(cfg.output_dir)

    logger.info(f"Done: {succeeded} processed, {failed} failed out of {len(unique_measurements)} measurements")


if __name__ == "__main__":
    main()
