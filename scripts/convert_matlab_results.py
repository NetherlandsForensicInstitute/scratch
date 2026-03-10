"""
Convert MATLAB result folders to Python by calling the preprocessor API.

Walks a nested folder structure, converts x3p files, extracts crop and
preprocessing parameters from .mat files, and calls the local preprocessor
API to regenerate marks.

Usage:
    python convert_matlab_results.py /path/to/root output/
    python convert_matlab_results.py /path/to/root output/ --api-url http://localhost:8000
"""

import argparse
import contextlib
import io
import json
import logging
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from parsers import load_scan_image, parse_to_x3p
from returns.unsafe import unsafe_perform_io
from tqdm import tqdm

from scripts.http_utils import _post_with_retry
from scripts.matlab_utils import (
    extract_impression_params,
    extract_mark_type,
    extract_mask_and_bounding_box,
    extract_striation_params,
    load_mat_struct,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Shared configuration for the conversion pipeline."""

    root: Path
    output_dir: Path
    api_url: str
    force: bool = False

    def __post_init__(self) -> None:
        self.api_url = self.api_url.rstrip("/")


def convert_x3p(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Load an X3P from path and parse it with the pipelines and write the result."""
    scan = unsafe_perform_io(load_scan_image(input_path).unwrap())
    x3p = parse_to_x3p(scan).unwrap()

    tmp = output_path.with_stem(f".{uuid.uuid4().hex}.tmp")
    try:
        x3p.write(str(tmp))
        os.replace(tmp, output_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise

    return scan.width, scan.height


def _save_shape(path: Path, shape: tuple[int, int]) -> None:
    path.with_suffix(".shape").write_text(f"{shape[0]},{shape[1]}")


def _load_shape(path: Path) -> tuple[int, int] | None:
    shape_file = path.with_suffix(".shape")
    if shape_file.exists():
        x, y = shape_file.read_text().strip().split(",")
        return int(x), int(y)
    return None


def find_mark_folders(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (measurement_folder, mark_folder) pairs found under root."""
    for mark_mat in root.rglob("mark.mat"):
        mf = mark_mat.parent
        if (mf.parent / "measurement.x3p").exists():
            yield mf.parent, mf


def copy_db_scratch_files(root: Path, output_dir: Path) -> int:
    """Copy all db.scratch files to output_dir, appending a conversion timestamp."""
    db_files = list(root.rglob("db.scratch"))
    for db_file in tqdm(db_files, desc="Copying db.scratch", unit=" files"):
        out = output_dir / db_file.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(db_file.read_text() + f"\nCONVERTED_DATE={datetime.now().isoformat()}\n")
    return len(db_files)


def convert_measurement_x3p(measurement_folder: Path, cfg: ConversionConfig) -> tuple[Path, tuple[int, int]]:
    """Convert a measurement.x3p and generate preview/surface_map images.

    :returns: (path to converted x3p, (size_x, size_y) pixel dimensions).
    """
    original = measurement_folder / "measurement.x3p"
    output_x3p = cfg.output_dir / measurement_folder.relative_to(cfg.root) / "measurement.x3p"

    if output_x3p.exists() and not cfg.force:
        shape = _load_shape(output_x3p)
        if shape is not None:
            return output_x3p, shape
        logger.warning("Missing shape file for %s, reconverting", output_x3p)

    output_x3p.parent.mkdir(parents=True, exist_ok=True)

    shape = convert_x3p(original, output_x3p)
    _save_shape(output_x3p, shape)

    result = _post_with_retry(f"{cfg.api_url}/preprocessor/process-scan", {"scan_file": str(output_x3p)})
    for key in ("preview", "surface_map"):
        if key in result:
            url = result[key]
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            (output_x3p.parent / url.rsplit("/", 1)[-1]).write_bytes(resp.content)

    return output_x3p, shape


def convert_mark(
    mark_folder: Path,
    converted_x3p: Path,
    shape: tuple[int, int],
    cfg: ConversionConfig,
) -> None:
    """Process a single mark: extract params, call API, download results.

    Reads crop info and preprocessing parameters from mark.mat, builds the
    API request as multipart form + file upload, and downloads the resulting
    files into the output directory.
    """
    mark_dir = cfg.output_dir / mark_folder.relative_to(cfg.root)

    if (mark_dir / "mark.json").exists() and not cfg.force:
        return

    struct = load_mat_struct(mark_folder / "mark.mat")
    mark_type = extract_mark_type(struct)

    size_x, size_y = shape
    mask, bounding_box_list = extract_mask_and_bounding_box(struct, size_x, size_y)

    is_impression = mark_type.is_impression()
    endpoint = f"preprocessor/prepare-mark-{'impression' if is_impression else 'striation'}"
    params = extract_impression_params(struct, mark_type) if is_impression else extract_striation_params(struct)

    # Build the JSON params (everything except the mask binary data)
    params_dict = {
        "scan_file": str(converted_x3p),
        "mark_type": mark_type.value,
        "mark_parameters": params,
        "bounding_box_list": bounding_box_list,
        "mask_is_bitpacked": False,
    }

    # Send as multipart: params as JSON form field, mask as file upload
    mask_bytes = mask.astype(np.bool_).tobytes()

    resp = requests.post(
        f"{cfg.api_url}/{endpoint}",
        data={"params": json.dumps(params_dict)},
        files={"mask_data": ("mask.bin", io.BytesIO(mask_bytes), "application/octet-stream")},
        timeout=300,
    )
    if resp.status_code == 422:
        logger.error("Validation error for %s: %s", mark_folder, resp.json())
    resp.raise_for_status()
    result = resp.json()

    processed_dir = mark_dir / "processed"
    mark_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mark_filenames = {"mark.npz", "mark.json"}
    for url in result.values():
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        filename = url.rsplit("/", 1)[-1]
        dest = mark_dir if filename in mark_filenames else processed_dir
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        (dest / filename).write_bytes(resp.content)


def _run_parallel(
    tasks: Iterable[tuple[Any, Callable, tuple]],
    workers: int,
    desc: str,
    unit: str,
) -> dict[Any, Any]:
    """Run tasks in parallel with a progress bar.

    :param tasks: iterable of (key, fn, args) tuples.
    :returns: dict of {key: result}.
    """
    task_list = list(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): key for key, fn, args in task_list}
        results = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit=unit):
            results[futures[future]] = future.result()
    return results


def main() -> None:
    """Entry point: parse args and run the conversion pipeline."""
    parser = argparse.ArgumentParser(description="Convert MATLAB results via Python API")
    parser.add_argument("root", type=Path, help="Root folder to search")
    parser.add_argument("output", type=Path, help="Output folder (mirrors input structure)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Preprocessor API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Reconvert even if output exists")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N marks (for debugging)")
    args = parser.parse_args()

    cfg = ConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    copy_db_scratch_files(cfg.root, cfg.output_dir)

    marks = list(tqdm(find_mark_folders(cfg.root), desc="Scanning", unit=" marks"))
    if args.limit:
        marks = marks[: args.limit]
    logger.info(f"Found {len(marks)} marks")

    unique_measurements = list({mf for mf, _ in marks})
    converted_x3ps = _run_parallel(
        ((mf, convert_measurement_x3p, (mf, cfg)) for mf in unique_measurements),
        args.workers,
        "Converting x3p",
        " files",
    )

    _run_parallel(
        ((mf, convert_mark, (mf, *converted_x3ps[meas], cfg)) for meas, mf in marks),
        args.workers,
        "Converting marks",
        " marks",
    )

    logger.info(f"Done: {len(marks)} marks, {len(converted_x3ps)} x3p files")


if __name__ == "__main__":
    main()
