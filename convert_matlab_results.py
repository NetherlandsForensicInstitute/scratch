"""Convert MATLAB result folders to Python by calling the preprocessor API.

Walks a nested folder structure, extracts crop and preprocessing parameters
from .mat files, and calls the local preprocessor API to regenerate results
from the original x3p files.

Usage:
    python convert_matlab_results.py /path/to/root output/ --api-url http://localhost:8000
    python convert_matlab_results.py /path/to/root output/ --dry-run
"""

import argparse
import logging
import xml.etree.ElementTree as ET  # noqa: S405
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import scipy.io as sio
from conversion.data_formats import MarkType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mark type mapping
# ---------------------------------------------------------------------------

MATLAB_MARK_TYPE_MAP: dict[str, MarkType] = {
    "Firing pin impression mark": MarkType.FIRING_PIN_IMPRESSION,
    "Breech face impression mark": MarkType.BREECH_FACE_IMPRESSION,
    "Chamber impression mark": MarkType.CHAMBER_IMPRESSION,
    "Ejector impression mark": MarkType.EJECTOR_IMPRESSION,
    "Extractor impression mark": MarkType.EXTRACTOR_IMPRESSION,
    "Aperture shear striation mark": MarkType.APERTURE_SHEAR_STRIATION,
    "Bullet GEA striation mark": MarkType.BULLET_GEA_STRIATION,
    "Bullet LEA striation mark": MarkType.BULLET_LEA_STRIATION,
    "Chamber striation mark": MarkType.CHAMBER_STRIATION,
    "Ejector striation mark": MarkType.EJECTOR_STRIATION,
    "Ejector port striation mark": MarkType.EJECTOR_PORT_STRIATION,
    "Extractor striation mark": MarkType.EXTRACTOR_STRIATION,
    "Firing pin drag striation mark": MarkType.FIRING_PIN_DRAG_STRIATION,
}


# ---------------------------------------------------------------------------
# Data extraction from .mat files
# ---------------------------------------------------------------------------


def _get_scalar(value: np.ndarray):
    """Unwrap a numpy scalar or 0-d array."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.flat[0]
    return value


def _get_struct_field(struct: np.ndarray, field: str):
    """Safely extract and unwrap a field from a MATLAB struct."""
    return _get_scalar(struct[field])


def _unwrap(value):
    """Recursively unwrap numpy scalar arrays."""
    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    return value


def _is_empty(value) -> bool:
    """Check if a value is an empty MATLAB array or None."""
    if value is None:
        return True
    if isinstance(value, np.ndarray) and value.size == 0:
        return True
    return False


def load_mat_struct(mat_path: Path) -> dict:
    """Load a .mat file and return the data_struct fields as a dict."""
    data = sio.loadmat(str(mat_path), squeeze_me=True)
    return data["data_struct"]


@dataclass
class CropDefinition:
    """Crop parameters extracted from a MATLAB mark.mat."""

    crop_type: str  # "ellipse" or "rectangle"
    # For ellipse: center (x, y), minor_axis, major_axis, angle
    # For rectangle: corners (4, 2) array in (x, y) = (col, row) order
    params: dict


def _parse_ellipse_params(raw_params) -> dict:
    """Parse ellipse crop parameters from MATLAB struct or tuple."""
    raw_params = _unwrap(raw_params)
    if isinstance(raw_params, tuple):
        return {
            "center": np.array(raw_params[0], dtype=np.float64),
            "minor_axis": float(raw_params[1]),
            "major_axis": float(raw_params[2]),
            "angle": float(raw_params[3]),
        }
    return {
        "center": np.array(_unwrap(raw_params["center"]), dtype=np.float64),
        "minor_axis": float(_unwrap(raw_params["minoraxis"])),
        "major_axis": float(_unwrap(raw_params["majoraxis"])),
        "angle": float(_unwrap(raw_params["angle_majoraxis"])),
    }


def _parse_circle_params(raw_params) -> dict:
    """Parse circle crop parameters, returning ellipse-compatible dict."""
    raw_params = _unwrap(raw_params)
    if isinstance(raw_params, tuple):
        center = np.array(raw_params[0], dtype=np.float64)
        radius = float(raw_params[1])
    else:
        center = np.array(_unwrap(raw_params["center"]), dtype=np.float64)
        radius = float(_unwrap(raw_params["radius"]))
    return {"center": center, "minor_axis": radius, "major_axis": radius, "angle": 0.0}


def _parse_rectangle_params(raw_params) -> dict:
    """Parse rectangle crop parameters from MATLAB struct or tuple."""
    raw_params = _unwrap(raw_params)
    if isinstance(raw_params, tuple):
        corners = np.array(raw_params[0], dtype=np.float64)
    elif isinstance(raw_params, np.ndarray) and raw_params.dtype.names and "corner" in raw_params.dtype.names:
        corners = _unwrap(raw_params["corner"]).astype(np.float64)
    else:
        corners = np.array(raw_params, dtype=np.float64)
    return {"corners": corners}


def _extract_crop_items(crop_info_raw: np.ndarray) -> list[np.ndarray] | None:
    """Extract the list of crop items from a raw crop_info array."""
    first = _unwrap(crop_info_raw[0])
    if isinstance(first, str):
        return [crop_info_raw]
    if isinstance(first, np.ndarray) and first.dtype == object:
        return [_unwrap(crop_info_raw[i]) for i in range(len(crop_info_raw))]
    logger.warning(f"  Unexpected crop_info[0] type: {type(first)}, skipping")
    return None


_CROP_PARSERS: dict[str, tuple[callable, str]] = {
    "ellipse": (_parse_ellipse_params, "ellipse"),
    "circle": (_parse_circle_params, "ellipse"),
    "rectangle": (_parse_rectangle_params, "rectangle"),
}


def extract_crop_info(struct) -> CropDefinition | None:
    """Extract crop definition from a MATLAB data_struct.

    crop_info may contain multiple items (e.g. a rectangle + circle).
    We use the first item for the bounding_box / mask definition.
    """
    crop_info_raw = _unwrap(struct["crop_info"])

    if isinstance(crop_info_raw, np.ndarray) and crop_info_raw.size == 0:
        return None

    # Unwrap single-element object array wrappers
    while isinstance(crop_info_raw, np.ndarray) and crop_info_raw.dtype == object and crop_info_raw.size == 1:
        crop_info_raw = crop_info_raw.flat[0]

    if not isinstance(crop_info_raw, np.ndarray) or crop_info_raw.dtype != object:
        logger.warning(f"  Unexpected crop_info format: {type(crop_info_raw)}, skipping")
        return None

    items = _extract_crop_items(crop_info_raw)
    if items is None:
        return None

    item = items[0]
    crop_type = str(_unwrap(item[0]))
    raw_params = _unwrap(item[1])

    if crop_type not in _CROP_PARSERS:
        raise ValueError(f"Unknown crop type: {crop_type}")

    parse_fn, resolved_type = _CROP_PARSERS[crop_type]
    return CropDefinition(crop_type=resolved_type, params=parse_fn(raw_params))


def extract_mark_type(struct) -> MarkType:
    """Extract and map the MATLAB mark type string to our MarkType enum."""
    matlab_type = str(_get_struct_field(struct, "mark_type"))
    if matlab_type not in MATLAB_MARK_TYPE_MAP:
        raise ValueError(f"Unknown MATLAB mark type: {matlab_type!r}")
    return MATLAB_MARK_TYPE_MAP[matlab_type]


def extract_data_param_field(struct, field: str, default=None):
    """Safely extract a field from the nested data_param struct."""
    data_param = struct["data_param"].item()
    if field in data_param.dtype.names:
        value = _get_scalar(data_param[field])
        if _is_empty(value):
            return default
        return value
    return default


def extract_impression_params(struct, mark_type: MarkType) -> dict:
    """Extract preprocessing parameters for impression marks."""
    cutoff_hi = _get_struct_field(struct, "cutoff_hi")
    cutoff_lo = _get_struct_field(struct, "cutoff_lo")

    return {
        "pixel_size": mark_type.scale,
        "adjust_pixel_spacing": bool(extract_data_param_field(struct, "bAdjustPixelSpacing", 1)),
        "level_offset": bool(extract_data_param_field(struct, "bLevelOffset", 1)),
        "level_tilt": bool(extract_data_param_field(struct, "bLevelTilt", 1)),
        "level_2nd": bool(extract_data_param_field(struct, "bLevel2nd", 1)),
        "interp_method": _interp_method(extract_data_param_field(struct, "intMeth", "cubic")),
        "highpass_cutoff": float(cutoff_hi) * 1e-6 if not _is_empty(cutoff_hi) else None,
        "lowpass_cutoff": float(cutoff_lo) * 1e-6 if not _is_empty(cutoff_lo) else None,
    }


def extract_striation_params(struct) -> dict:
    """Extract preprocessing parameters for striation marks."""
    cutoff_hi = _get_struct_field(struct, "cutoff_hi")
    cutoff_lo = _get_struct_field(struct, "cutoff_lo")

    return {
        "highpass_cutoff": float(cutoff_hi) * 1e-6 if not _is_empty(cutoff_hi) else 2e-3,
        "lowpass_cutoff": float(cutoff_lo) * 1e-6 if not _is_empty(cutoff_lo) else 2.5e-4,
        "cut_borders_after_smoothing": bool(extract_data_param_field(struct, "cut_borders_after_smoothing", 1)),
        "use_mean": bool(extract_data_param_field(struct, "use_mean", 1)),
        "angle_accuracy": float(extract_data_param_field(struct, "angle_accuracy", 0.1)),
        "subsampling_factor": int(_get_struct_field(struct, "subsampling") or 1),
    }


def _interp_method(value) -> str:
    """Map MATLAB interpolation method to Python equivalent."""
    mapping = {"cubic": "cubic", "linear": "linear", "nearest": "nearest"}
    if isinstance(value, str):
        return mapping.get(value.lower(), "cubic")
    return "cubic"


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


def get_x3p_shape(x3p_path: Path) -> tuple[int, int]:
    """Read x3p pixel dimensions (SizeX, SizeY) from its XML header."""
    with zipfile.ZipFile(x3p_path) as z:
        with z.open("main.xml") as f:
            tree = ET.parse(f)  # noqa: S314
        root = tree.getroot()
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        sx = int(root.find(".//MatrixDimension/SizeX").text)
        sy = int(root.find(".//MatrixDimension/SizeY").text)

    return sx, sy


def generate_mask(
    crop: CropDefinition,
    size_x: int,
    size_y: int,
) -> np.ndarray:
    """Generate a boolean mask on the x3p grid.

    The x3p grid has shape (SizeY, SizeX) when loaded as a 2D array
    (rows=Y, cols=X). Crop coordinates are in (x, y) = (col, row) order.

    Returns a (SizeY, SizeX) boolean array.
    """
    if crop.crop_type == "ellipse":
        cx, cy = crop.params["center"]
        minor = crop.params["minor_axis"]
        major = crop.params["major_axis"]
        angle_deg = crop.params["angle"]

        row, col = np.ogrid[:size_y, :size_x]
        dx = col - cx
        dy = row - cy

        cos_a = np.cos(np.radians(angle_deg))
        sin_a = np.sin(np.radians(angle_deg))
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy

        return (xr / major) ** 2 + (yr / minor) ** 2 <= 1.0

    if crop.crop_type == "rectangle":
        corners = crop.params["corners"]
        x_min = int(corners[:, 0].min())
        x_max = int(corners[:, 0].max())
        y_min = int(corners[:, 1].min())
        y_max = int(corners[:, 1].max())

        mask = np.zeros((size_y, size_x), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        return mask

    raise ValueError(f"Unknown crop type: {crop.crop_type}")


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------


def build_request_body(
    scan_file: Path,
    mark_type: MarkType,
    mask: np.ndarray,
    bounding_box: np.ndarray | None,
    preprocess_params: dict,
) -> dict:
    """Build the JSON request body for the preprocessor API."""
    body = {
        "scan_file": str(scan_file),
        "mark_type": mark_type.value,
        "mask_array": mask.astype(float).tolist(),
        "mark_parameters": preprocess_params,
    }

    if bounding_box is not None:
        body["bounding_box"] = bounding_box.tolist()

    return body


def call_api(api_url: str, endpoint: str, body: dict) -> dict:
    """POST to the preprocessor API and return the response."""
    url = f"{api_url.rstrip('/')}/{endpoint}"
    response = requests.post(url, json=body, timeout=300)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Folder traversal and conversion
# ---------------------------------------------------------------------------


def find_mark_folders(root: Path):
    """Yield (measurement_folder, mark_folder) pairs."""
    for mark_mat in root.rglob("mark.mat"):
        mark_folder = mark_mat.parent
        measurement_folder = mark_folder.parent

        x3p = measurement_folder / "measurement.x3p"
        if not x3p.exists():
            logger.warning(f"No measurement.x3p found for {mark_folder}, skipping")
            continue

        yield measurement_folder, mark_folder


@dataclass
class ConversionTask:
    """Parameters for a single mark conversion."""

    root: Path
    measurement_folder: Path
    mark_folder: Path
    output_dir: Path
    api_url: str
    dry_run: bool = False


def convert_mark(task: ConversionTask) -> None:
    """Convert a single MATLAB mark result by calling the Python API."""
    x3p_path = task.measurement_folder / "measurement.x3p"
    mark_mat_path = task.mark_folder / "mark.mat"

    logger.info(f"Processing: {task.mark_folder}")

    struct = load_mat_struct(mark_mat_path)
    mark_type = extract_mark_type(struct)
    is_impression = mark_type.is_impression()

    crop = extract_crop_info(struct)
    if crop is None:
        logger.warning("  No crop info found, skipping")
        return

    size_x, size_y = get_x3p_shape(x3p_path)
    mask = generate_mask(crop, size_x, size_y)
    bounding_box = crop.params.get("corners") if crop.crop_type == "rectangle" else None

    if is_impression:
        preprocess_params = extract_impression_params(struct, mark_type)
        endpoint = "preprocessor/prepare-mark-impression"
    else:
        preprocess_params = extract_striation_params(struct)
        endpoint = "preprocessor/prepare-mark-striation"

    relative_path = task.mark_folder.relative_to(task.root)
    output_mark_dir = task.output_dir / relative_path
    output_mark_dir.mkdir(parents=True, exist_ok=True)

    body = build_request_body(
        scan_file=x3p_path,
        mark_type=mark_type,
        mask=mask,
        bounding_box=bounding_box,
        preprocess_params=preprocess_params,
    )

    if task.dry_run:
        logger.info(f"  mark_type: {mark_type}")
        logger.info(f"  crop_type: {crop.crop_type}")
        logger.info(f"  mask shape: {mask.shape}, True pixels: {mask.sum()}")
        logger.info(f"  bounding_box: {'(4,2) array' if bounding_box is not None else None}")
        logger.info(f"  endpoint: {endpoint}")
        logger.info(f"  params: {preprocess_params}")
        logger.info(f"  output: {output_mark_dir}")
        return

    try:
        debug_body = {k: v for k, v in body.items() if k != "mask_array"}
        debug_body["mask_array"] = f"<{len(body['mask_array'])}x{len(body['mask_array'][0])} array>"
        logger.debug(f"  Request: {debug_body}")

        result = call_api(task.api_url, endpoint, body)
        logger.info(f"  Success: {result}")
        save_results(result, output_mark_dir)
    except requests.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"  API call failed: {e}")
            logger.error(f"  Response body: {e.response.text}")
        else:
            logger.error(f"  API call failed: {e}")


def save_results(api_response: dict, output_dir: Path) -> None:
    """Download result files from the API response URLs and save them to output_dir."""
    for _field_name, url in api_response.items():
        if not isinstance(url, str) or not url.startswith("http"):
            continue

        filename = url.rsplit("/", 1)[-1]
        output_path = output_dir / filename

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        logger.info(f"  Saved {filename}")


def main():
    """Convert MATLAB result folders by calling the Python preprocessor API."""
    parser = argparse.ArgumentParser(description="Convert MATLAB results via Python API")
    parser.add_argument("root", type=Path, help="Root folder to search")
    parser.add_argument("output", type=Path, help="Output folder (mirrors input structure)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Preprocessor API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without calling API")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    count = 0
    for measurement_folder, mark_folder in find_mark_folders(args.root):
        count += 1
        logger.info(f"[{count}] {mark_folder}")
        task = ConversionTask(
            root=args.root,
            measurement_folder=measurement_folder,
            mark_folder=mark_folder,
            output_dir=args.output,
            api_url=args.api_url,
            dry_run=args.dry_run,
        )
        convert_mark(task)

    logger.info(f"Processed {count} marks")


if __name__ == "__main__":
    main()
