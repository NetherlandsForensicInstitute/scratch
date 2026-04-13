"""
Fetch marks from the preprocessor API using already-converted x3p files.

Expects that convert_x3ps.py has already been run, so that converted
measurement.x3p and .shape sidecar files exist in the output directory.
"""

import argparse
import io
import json
import logging
import queue
import threading
from pathlib import Path

import numpy as np
import requests
from conversion.data_formats import MarkImpressionType
from tqdm import tqdm

from scripts.conversion_utils import ConversionConfig, find_mark_folders, load_shape, run_parallel
from scripts.http_utils import _cleanup_vault, _post_with_retry
from scripts.matlab_utils import (
    extract_impression_params,
    extract_mark_type,
    extract_mask_and_bounding_box,
    extract_striation_params,
    load_mat_struct,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_converted_x3ps(
    marks: list[tuple[Path, Path]],
    cfg: ConversionConfig,
) -> dict[Path, tuple[Path, tuple[int, int]]]:
    """Build a mapping from measurement folder to (converted x3p path, shape).

    Raises if a required .shape sidecar is missing (run convert_x3ps.py first).
    """
    converted: dict[Path, tuple[Path, tuple[int, int]]] = {}
    for meas_folder, _ in marks:
        if meas_folder in converted:
            continue
        output_x3p = cfg.output_dir / meas_folder.relative_to(cfg.root) / "measurement.x3p"
        if not output_x3p.exists():
            raise FileNotFoundError(f"Converted x3p not found: {output_x3p}. Run convert_x3ps.py first.")
        shape = load_shape(output_x3p)
        if shape is None:
            raise FileNotFoundError(f"Shape sidecar not found for {output_x3p}. Run convert_x3ps.py first.")
        converted[meas_folder] = (output_x3p, shape)
    return converted


def _download_result_files(result: dict[str, object], session: requests.Session) -> dict[str, bytes]:
    """Download all file URLs from an API result dict.

    Uses the provided session for connection reuse. Cleans up vault entries
    even if a download fails partway through.
    """
    try:
        downloaded = {}
        for url in result.values():
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            filename = url.rsplit("/", 1)[-1]
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            downloaded[filename] = resp.content
        return downloaded
    finally:
        _cleanup_vault(result)


def fetch_mark(
    mark_folder: Path,
    converted_x3p: Path,
    shape: tuple[int, int],
    cfg: ConversionConfig,
    session: requests.Session,
) -> tuple[Path, dict[str, bytes]] | None:
    """Call the API and download results into memory.

    Returns None when the mark was already converted and --force is not set.
    """
    mark_dir = cfg.output_dir / mark_folder.relative_to(cfg.root)

    if (mark_dir / "mark.json").exists() and not cfg.force:
        return None

    struct = load_mat_struct(mark_folder / "mark.mat")
    mark_type = extract_mark_type(struct)

    size_x, size_y = shape
    mask, bounding_box_list = extract_mask_and_bounding_box(struct, size_x, size_y)

    is_impression = isinstance(mark_type, MarkImpressionType)
    endpoint = f"preprocessor/prepare-mark-{'impression' if is_impression else 'striation'}"
    params = extract_impression_params(struct, mark_type) if is_impression else extract_striation_params(struct)

    params_dict = {
        "scan_file": str(converted_x3p),
        "mark_type": mark_type.value,
        "mark_parameters": params,
        "bounding_box_list": bounding_box_list,
        "mask_is_bitpacked": False,
    }

    mask_bytes = mask.astype(np.bool_).tobytes()
    result = _post_with_retry(
        f"{cfg.api_url}/{endpoint}",
        data={"params": json.dumps(params_dict)},
        files={"mask_data": ("mask.bin", io.BytesIO(mask_bytes), "application/octet-stream")},
    )

    downloaded = _download_result_files(result, session)
    return mark_folder, downloaded


def _write_mark(mark_folder: Path, files: dict[str, bytes], cfg: ConversionConfig) -> None:
    """Write fetched mark data to disk (called from the single writer thread)."""
    mark_dir = cfg.output_dir / mark_folder.relative_to(cfg.root)
    mark_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (mark_dir / filename).write_bytes(content)


def _disk_writer(
    result_queue: queue.Queue[tuple[Path, dict[str, bytes]] | None],
    cfg: ConversionConfig,
    progress: tqdm,
) -> None:
    """Single-threaded consumer that serialises all disk writes."""
    while True:
        item = result_queue.get()
        if item is None:
            break
        mark_folder, files = item
        _write_mark(mark_folder, files, cfg)
        progress.update(1)


def _fetch_and_enqueue(
    result_queue: queue.Queue[tuple[Path, dict[str, bytes]] | None],
    progress: tqdm,
    mark_folder: Path,
    converted_x3p: Path,
    shape: tuple[int, int],
    cfg: ConversionConfig,
    session: requests.Session,
) -> None:
    """
    Fetch a mark and put the result on the write queue.

    Logs and continues on failure so that one bad mark doesn't abort the batch.
    Always updates *progress* so the bar reflects skipped and failed marks too.
    """
    try:
        mark_result = fetch_mark(mark_folder, converted_x3p, shape, cfg, session)
        if mark_result is not None:
            result_queue.put(mark_result)
        else:
            progress.update(1)
    except Exception:
        logger.exception("Failed to fetch mark %s", mark_folder)
        progress.update(1)


def convert_marks_parallel(
    marks: list[tuple[Path, Path]],
    converted_x3ps: dict[Path, tuple[Path, tuple[int, int]]],
    cfg: ConversionConfig,
    workers: int,
) -> None:
    """Fetch marks in parallel, write results sequentially via a single writer thread."""
    write_queue = queue.Queue(maxsize=workers * 2)

    progress = tqdm(total=len(marks), desc="Converting marks", unit=" marks")
    writer = threading.Thread(target=_disk_writer, args=(write_queue, cfg, progress), daemon=True)
    writer.start()

    session = requests.Session()
    try:
        run_parallel(
            (
                (mf, _fetch_and_enqueue, (write_queue, progress, mf, *converted_x3ps[meas], cfg, session))
                for meas, mf in marks
            ),
            workers,
            "Fetching marks",
            " marks",
        )
    finally:
        write_queue.put(None)
        writer.join()
        progress.close()
        session.close()


def main() -> None:
    """Entry point: fetch all marks using previously converted x3p files."""
    parser = argparse.ArgumentParser(description="Fetch marks via the preprocessor API")
    parser.add_argument("root", type=Path, help="Root folder to search (original MATLAB results)")
    parser.add_argument("output", type=Path, help="Output folder (must contain converted x3ps)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Preprocessor API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Reconvert even if output exists")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N marks (for debugging)")
    args = parser.parse_args()

    cfg = ConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)

    if not cfg.output_dir.exists():
        raise FileNotFoundError(f"Output directory {cfg.output_dir} does not exist. Run convert_x3ps.py first.")

    marks = list(tqdm(find_mark_folders(cfg.root), desc="Scanning", unit=" marks"))
    if args.limit:
        marks = marks[: args.limit]
    logger.info(f"Found {len(marks)} marks")

    converted_x3ps = _resolve_converted_x3ps(marks, cfg)
    logger.info(f"Resolved {len(converted_x3ps)} converted x3p files")

    convert_marks_parallel(marks, converted_x3ps, cfg, args.workers)

    logger.info(f"Done: {len(marks)} marks processed")


if __name__ == "__main__":
    main()
