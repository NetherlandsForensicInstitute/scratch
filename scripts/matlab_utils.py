import logging
from pathlib import Path
from typing import Any

import numpy as np
from conversion.data_formats import MarkImpressionType, MarkStriationType, MarkType
from scipy import io as sio
from scipy.constants import micro

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


def _parse_crop_item(crop_item: np.ndarray, size_x: int, size_y: int) -> tuple[np.ndarray, list | None]:
    """Parse the crop item."""
    crop_type = str(_scalar(crop_item[0]))
    raw_params = _scalar(crop_item[1])

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
        return mask, corners.tolist()

    raise ValueError(f"Unknown crop type: {crop_type}")


def extract_mask_and_bounding_box(
    struct: np.ndarray,
    size_x: int,
    size_y: int,
) -> tuple[np.ndarray, list | None]:
    """Extract a boolean mask and optional bounding box from crop_info in a MATLAB struct.

    Uses the first crop item. For ellipse/circle crops the bounding_box is None.
    For rectangle crops the bounding_box is a (4, 2) float corners array.

    :returns: (mask, bounding_box) or None if no valid crop info found.
    """
    crop_raw = _scalar(struct["crop_info"])
    while isinstance(crop_raw, np.ndarray) and crop_raw.dtype == object and crop_raw.size == 1:
        crop_raw = crop_raw.flat[0]

    # Handle both (3,) single crop and (N, 3) multiple crops
    if crop_raw.ndim == 2:  # noqa: PLR2004
        crop_items = [crop_raw[i] for i in range(crop_raw.shape[0])]
    else:
        crop_items = [crop_raw]

    # Try each crop item, return the first that produces a non-empty mask
    for crop_item in crop_items:
        mask, bbox = _parse_crop_item(crop_item, size_x, size_y)
        if mask.any():
            return mask, bbox

    logger.warning("All crop items produced empty masks")
    raise ValueError("All crop items produced empty masks")


def extract_mark_type(struct: np.ndarray) -> MarkType:
    """Extract the marktype, impression or striation."""
    mark_string = str(_scalar(struct["mark_type"])).lower()
    try:
        return MarkImpressionType(mark_string)
    except ValueError:
        return MarkStriationType(mark_string)


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
        "highpass_cutoff": float(hi) * 1e-6 if hi is not None else 400.0 * micro,
        "lowpass_cutoff": float(lo) * 1e-6 if lo is not None else 25 * micro,
    }


def extract_striation_params(struct: np.ndarray) -> dict[str, Any]:
    """Extract preprocessing parameters for striation marks from a MATLAB struct."""
    hi, lo = _scalar(struct["cutoff_hi"]), _scalar(struct["cutoff_lo"])
    return {
        "highpass_cutoff": float(hi) * 1e-6 if hi is not None else 250e-6,
        "lowpass_cutoff": float(lo) * 1e-6 if lo is not None else 5e-6,
        "cut_borders_after_smoothing": bool(_data_param_field(struct, "cut_borders_after_smoothing", 1)),
        "use_mean": bool(_data_param_field(struct, "use_mean", 1)),
        "angle_accuracy": float(_data_param_field(struct, "angle_accuracy", 90)),
        "subsampling_factor": int(_scalar(struct["subsampling"]) or 1),
    }


def unwrap_path(value: Any) -> str | None:
    """Recursively unwrap a MATLAB path value to a plain string.

    MATLAB paths from ``loadmat`` may arrive as nested object arrays,
    e.g. ``array([array(['tool-entries/...'], dtype='<U64')])``.

    :param value: raw value from loadmat.
    :returns: plain string or ``None`` if empty.
    """
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    return str(value) if value is not None else None
