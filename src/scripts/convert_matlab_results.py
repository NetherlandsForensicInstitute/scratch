"""
Convert MATLAB result folders to Python by calling the preprocessor API.

Walks a nested folder structure, converts x3p files (layer swap + checksum fix),
extracts crop and preprocessing parameters from .mat files, and calls the local
preprocessor API to regenerate marks.

Usage:
    python convert_matlab_results.py /path/to/root output/
    python convert_matlab_results.py /path/to/root output/ --api-url http://localhost:8000
"""

import argparse
import hashlib
import logging
import xml.etree.ElementTree as ET  # noqa: S405
import zipfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
import scipy.io as sio
from conversion.data_formats import MarkType
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_MIN_LAYERS_FOR_SWAP = 2


@dataclass
class ConversionConfig:
    """Shared configuration for the conversion pipeline."""

    root: Path
    output_dir: Path
    api_url: str
    force: bool = False

    def __post_init__(self) -> None:
        self.api_url = self.api_url.rstrip("/")


def _scalar(value: Any) -> Any:
    """Unwrap numpy 0-d arrays, single-element arrays, and empty arrays (→ None)."""
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.flat[0]
        else:
            break
    return value


def _data_param_field(struct: np.ndarray, field: str, default: Any = None) -> Any:
    """Extract a field from the nested data_param sub-struct."""
    dp = struct["data_param"].item()
    if field in dp.dtype.names:
        val = _scalar(dp[field])
        return default if val is None else val
    return default


def load_mat_struct(mat_path: Path) -> np.ndarray:
    """Load a .mat file and return the data_struct."""
    return sio.loadmat(str(mat_path), squeeze_me=True)["data_struct"]


def _local_tag(el: ET.Element) -> str:
    """Strip XML namespace prefix from an element tag."""
    return el.tag.split("}")[-1] if "}" in el.tag else el.tag


def _x3p_metadata(root: ET.Element) -> tuple[int, int, np.dtype]:
    """Extract (SizeX, SizeY, z_dtype) from parsed x3p XML in a single pass."""
    nx = ny = 0
    z_dtype = np.dtype(np.float32)
    dtype_map = {"F": np.dtype(np.float32), "D": np.dtype(np.float64), "I": np.dtype(np.int16), "L": np.dtype(np.int32)}
    for el in root.iter():
        tag = _local_tag(el)
        if tag == "SizeX" and el.text:
            nx = int(el.text.strip())
        elif tag == "SizeY" and el.text:
            ny = int(el.text.strip())
        elif tag == "CZ":
            for child in el:
                if _local_tag(child) == "DataType" and child.text:
                    dt = child.text.strip().upper()
                    if dt in dtype_map:
                        z_dtype = dtype_map[dt]
    return nx, ny, z_dtype


def convert_x3p(input_path: Path, output_path: Path) -> None:
    """
    Convert a MATLAB x3p to Python format by swapping data layers and fixing checksums.

    MATLAB x3p files store height in the second layer; Python expects it in the first.
    Also recomputes MD5 checksums for data.bin and main.xml.
    """
    with zipfile.ZipFile(input_path, "r") as zf_in:
        xml_bytes = zf_in.read("main.xml")
        tree = ET.parse(BytesIO(xml_bytes))  # noqa: S314
        root = tree.getroot()
        if root is None:
            raise ValueError(f"Could not parse main.xml in {input_path}")
        nx, ny, z_dtype = _x3p_metadata(root)

        if nx == 0 or ny == 0:
            raise ValueError(f"Could not determine dimensions (nx={nx}, ny={ny}) in {input_path}")

        data_path = next((dp for dp in ["bindata/data.bin", "data.bin"] if dp in zf_in.namelist()), None)
        if data_path is None:
            raise ValueError(f"No data.bin found in {input_path}")

        data = np.frombuffer(zf_in.read(data_path), dtype=z_dtype).copy()
        expected = nx * ny
        n_layers = len(data) // expected

        if n_layers < 1:
            raise ValueError(f"Data size mismatch in {input_path}: {len(data)} vs expected multiple of {expected}")

        layers = [data[i * expected : (i + 1) * expected] for i in range(n_layers)]
        if n_layers >= _MIN_LAYERS_FOR_SWAP:
            layers = layers[::-1]
        new_raw = np.concatenate(layers).astype(z_dtype).tobytes()

        data_md5 = hashlib.md5(new_raw).hexdigest()  # noqa: S324
        for el in root.iter():
            if _local_tag(el) == "MD5ChecksumPointData":
                el.text = data_md5

        xml_out = BytesIO()
        ET.ElementTree(root).write(xml_out, xml_declaration=True, encoding="UTF-8")
        xml_new = xml_out.getvalue()
        xml_md5 = hashlib.md5(xml_new).hexdigest()  # noqa: S324

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
            for item in zf_in.namelist():
                if item == data_path:
                    zf_out.writestr(item, new_raw)
                elif item == "main.xml":
                    zf_out.writestr(item, xml_new)
                elif item == "md5checksum.hex":
                    zf_out.writestr(item, xml_md5)
                else:
                    zf_out.writestr(item, zf_in.read(item))


def get_x3p_shape(x3p_path: Path) -> tuple[int, int]:
    """Read pixel dimensions (SizeX, SizeY) from an x3p file's XML header."""
    with zipfile.ZipFile(x3p_path) as z:
        tree = ET.parse(z.open("main.xml"))  # noqa: S314
    root = tree.getroot()
    if root is None:
        raise ValueError(f"Could not parse main.xml in {x3p_path}")
    nx, ny, _ = _x3p_metadata(root)
    if nx == 0 or ny == 0:
        raise ValueError(f"Missing SizeX/SizeY in {x3p_path}")
    return nx, ny


def _parse_ellipse(raw: Any) -> dict[str, Any]:
    """Parse ellipse crop parameters from a MATLAB tuple."""
    raw = _scalar(raw)
    return {
        "center": np.asarray(raw[0], dtype=float),
        "minor": float(raw[1]),
        "major": float(raw[2]),
        "angle": float(raw[3]),
    }


def _parse_circle(raw: Any) -> dict[str, Any]:
    """Parse circle crop parameters, returning ellipse-compatible dict."""
    raw = _scalar(raw)
    center, radius = np.asarray(raw[0], dtype=float), float(raw[1])
    return {"center": center, "minor": radius, "major": radius, "angle": 0.0}


def _parse_rectangle(raw: Any) -> np.ndarray:
    """Parse rectangle crop parameters, returning (4, 2) corner array."""
    raw = _scalar(raw)
    return np.asarray(raw[0], dtype=float)


def extract_mask_and_bounding_box(
    struct: np.ndarray,
    size_x: int,
    size_y: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract a boolean mask and optional bounding box from crop_info in a MATLAB struct.

    Uses the first crop item. For ellipse/circle crops the bounding_box is None.
    For rectangle crops the bounding_box is a (4, 2) float corners array.

    :returns: (mask, bounding_box) or None if no valid crop info found.
    """
    crop_raw = _scalar(struct["crop_info"])
    while isinstance(crop_raw, np.ndarray) and crop_raw.dtype == object and crop_raw.size == 1:
        crop_raw = crop_raw.flat[0]

    crop_type = str(_scalar(crop_raw[0]))
    raw_params = _scalar(crop_raw[1])

    if crop_type in ("ellipse", "circle"):
        p = _parse_circle(raw_params) if crop_type == "circle" else _parse_ellipse(raw_params)
        row, col = np.ogrid[:size_y, :size_x]
        cos_a, sin_a = np.cos(np.radians(p["angle"])), np.sin(np.radians(p["angle"]))
        dx, dy = col - p["center"][0], row - p["center"][1]
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy
        mask = (xr / p["major"]) ** 2 + (yr / p["minor"]) ** 2 <= 1.0
        return np.ascontiguousarray(mask[::-1, :]), None

    if crop_type == "rectangle":
        corners = _parse_rectangle(raw_params)
        corners[:, 1] = size_y - corners[:, 1]
        x0, x1 = int(corners[:, 0].min()), int(corners[:, 0].max())
        y0, y1 = int(corners[:, 1].min()), int(corners[:, 1].max())
        mask = np.zeros((size_y, size_x), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask, corners

    raise ValueError(f"Unknown crop type: {crop_type}")


def extract_mark_type(struct: np.ndarray) -> MarkType:
    """Extract and map the MATLAB mark type string to a MarkType enum."""
    return MarkType(str(_scalar(struct["mark_type"])).lower())


def extract_impression_params(struct: np.ndarray, mark_type: MarkType) -> dict[str, Any]:
    """Extract preprocessing parameters for impression marks from a MATLAB struct."""
    hi, lo = _scalar(struct["cutoff_hi"]), _scalar(struct["cutoff_lo"])
    return {
        "pixel_size": mark_type.scale,
        "adjust_pixel_spacing": bool(_data_param_field(struct, "bAdjustPixelSpacing", 1)),
        "level_offset": bool(_data_param_field(struct, "bLevelOffset", 1)),
        "level_tilt": bool(_data_param_field(struct, "bLevelTilt", 1)),
        "level_2nd": bool(_data_param_field(struct, "bLevel2nd", 1)),
        "interp_method": _data_param_field(struct, "intMeth", "cubic"),
        "highpass_cutoff": float(hi) * 1e-6 if hi is not None else None,
        "lowpass_cutoff": float(lo) * 1e-6 if lo is not None else None,
    }


def extract_striation_params(struct: np.ndarray) -> dict[str, Any]:
    """Extract preprocessing parameters for striation marks from a MATLAB struct."""
    hi, lo = _scalar(struct["cutoff_hi"]), _scalar(struct["cutoff_lo"])
    return {
        "highpass_cutoff": float(hi) * 1e-6 if hi is not None else 2e-3,
        "lowpass_cutoff": float(lo) * 1e-6 if lo is not None else 2.5e-4,
        "cut_borders_after_smoothing": bool(_data_param_field(struct, "cut_borders_after_smoothing", 1)),
        "use_mean": bool(_data_param_field(struct, "use_mean", 1)),
        "angle_accuracy": float(_data_param_field(struct, "angle_accuracy", 0.1)),
        "subsampling_factor": int(_scalar(struct["subsampling"]) or 1),
    }


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


def convert_measurement_x3p(measurement_folder: Path, cfg: ConversionConfig) -> Path:
    """Convert a measurement.x3p and generate preview/surface_map images.

    :returns: Path to the converted x3p in the output directory.
    """
    original = measurement_folder / "measurement.x3p"
    output_x3p = cfg.output_dir / measurement_folder.relative_to(cfg.root) / "measurement.x3p"

    if output_x3p.exists() and not cfg.force:
        return output_x3p

    output_x3p.parent.mkdir(parents=True, exist_ok=True)

    convert_x3p(original, output_x3p)

    result = requests.post(
        f"{cfg.api_url}/preprocessor/process-scan",
        json={"scan_file": str(output_x3p)},
        timeout=300,
    )
    result.raise_for_status()
    result = result.json()
    for key in ("preview", "surface_map"):
        if key in result:
            url = result[key]
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            (output_x3p.parent / url.rsplit("/", 1)[-1]).write_bytes(resp.content)

    return output_x3p


def convert_mark(mark_folder: Path, converted_x3p: Path, cfg: ConversionConfig) -> None:
    """Process a single mark: extract params, call API, download results.

    Reads crop info and preprocessing parameters from mark.mat, builds the
    API request, and downloads the resulting files into the output directory.
    """
    measurement_folder = mark_folder.parent
    mark_dir = cfg.output_dir / mark_folder.relative_to(cfg.root)

    if (mark_dir / "mark.json").exists() and not cfg.force:
        return

    struct = load_mat_struct(mark_folder / "mark.mat")
    mark_type = extract_mark_type(struct)

    size_x, size_y = get_x3p_shape(measurement_folder / "measurement.x3p")
    crop_result = extract_mask_and_bounding_box(struct, size_x, size_y)
    if crop_result is None:
        raise RuntimeError(f"No crop info in {mark_folder / 'mark.mat'}")
    mask, bounding_box = crop_result

    is_impression = mark_type.is_impression()
    endpoint = f"preprocessor/prepare-mark-{'impression' if is_impression else 'striation'}"
    params = extract_impression_params(struct, mark_type) if is_impression else extract_striation_params(struct)

    body: dict[str, Any] = {
        "scan_file": str(converted_x3p),
        "mark_type": mark_type.value,
        "mask": mask.astype(float).tolist() if mask else None,
        "mark_parameters": params,
    }
    if bounding_box is not None:
        body["bounding_box_list"] = bounding_box.tolist()

    resp = requests.post(f"{cfg.api_url}/{endpoint}", json=body, timeout=300)
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
    args = parser.parse_args()

    cfg = ConversionConfig(root=args.root, output_dir=args.output, api_url=args.api_url, force=args.force)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    copy_db_scratch_files(cfg.root, cfg.output_dir)

    marks = list(tqdm(find_mark_folders(cfg.root), desc="Scanning", unit=" marks"))
    logger.info(f"Found {len(marks)} marks")

    unique_measurements = list({mf for mf, _ in marks})
    converted_x3ps = _run_parallel(
        ((mf, convert_measurement_x3p, (mf, cfg)) for mf in unique_measurements),
        args.workers,
        "Converting x3p",
        " files",
    )

    _run_parallel(
        ((mf, convert_mark, (mf, converted_x3ps[meas], cfg)) for meas, mf in marks),
        args.workers,
        "Converting marks",
        " marks",
    )

    logger.info(f"Done: {len(marks)} marks, {len(converted_x3ps)} x3p files")


if __name__ == "__main__":
    main()
