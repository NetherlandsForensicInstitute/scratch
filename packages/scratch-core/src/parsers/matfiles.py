from pathlib import Path
from typing import Any, Type

import numpy as np

from scipy.io import loadmat

from models.enums import CropType, ImageType
from models.image import ImageData


def _is_empty(value: Any) -> bool:
    """Check if a value is empty (None, empty array, or empty list)."""
    return (
        value is None
        or (isinstance(value, np.ndarray) and not value.size)
        or (isinstance(value, (list, str)) and not value)
    )


def _extract_string_from_mat_field(value: Any) -> str:
    """Extract string value from MAT file field (handles numpy arrays)."""
    if isinstance(value, np.ndarray):
        return str(value.flat[0]) if value.size else ""
    return str(value) if value else ""


def _should_use_select_coordinates(select_coords: Any, crop_coords: Any) -> bool:
    """Determine if select_coordinates should replace crop_coordinates."""
    # If select is empty, keep crop as is
    if _is_empty(select_coords):
        return False

    # If crop is empty but select isn't, use select
    if _is_empty(crop_coords):
        return True

    # Both non-empty: compare them
    # Use select if arrays differ
    return not np.array_equal(
        np.asarray(crop_coords).flatten(),
        np.asarray(select_coords).flatten(),
    )


def _rename_selection_to_crop(
    selection_type: Any, select_coordinates: Any, **_
) -> dict[str, Any]:
    """Copy selection_type and select_coordinates to crop fields."""
    return {
        "crop_type": selection_type,
        "crop_coordinates": select_coordinates,
    }


def migrate_scratch_2_to_3_fields(mat_data: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate Scratch 2.0 field names to Scratch 3.0 format.

    In Scratch 2.0, cropping used 'selection_type' and 'select_coordinates'.
    In Scratch 3.0, these were renamed to 'crop_type' and 'crop_coordinates'.
    This function handles backward compatibility by migrating old fields to new ones.

    Parameters
    ----------
    mat_data : dict
        Dictionary loaded from MAT file

    Returns
    -------
    dict
        Modified dictionary with migrated fields
    """
    # Work with a copy to avoid mutating the input
    data = mat_data.copy()

    selection_type = data.pop("selection_type", "")
    select_coordinates = data.pop("select_coordinates", [])
    crop_type = data.pop("crop_type", "")
    crop_coordinates = data.pop("crop_coordinates", [])

    # Case 1: Both field sets exist (data from Scratch 3.0 or 2.0 opened in 3.0)
    if selection_type and crop_type:
        selection_type = _extract_string_from_mat_field(selection_type)
        crop_type = _extract_string_from_mat_field(crop_type)

        # If types differ, this is Scratch 2.0 data that was re-cropped
        if selection_type != crop_type:
            return data | _rename_selection_to_crop(selection_type, select_coordinates)

        # Types match (Scratch 3.0): synchronize coordinates if needed
        coords = (
            select_coordinates
            if _should_use_select_coordinates(select_coordinates, crop_coordinates)
            else crop_coordinates
        )
        return data | {"crop_type": crop_type, "crop_coordinates": coords}

    # Case 2: Only selection exists - migrate from Scratch 2.0
    if selection_type:
        return data | _rename_selection_to_crop(selection_type, select_coordinates)

    # Case 3: Neither exists - initialize as empty
    return data


def _cast_mat_value[T](type_: Type[T], value: Any) -> T:
    if type_ is str:
        return type_(
            value.flat[0]  # Handle both scalar and array cases
            if isinstance(value, np.ndarray)
            else value or ""
        )
    if type_ is float:
        return type_(value.item() if isinstance(value, np.ndarray) else value or 0.0)
    raise NotImplementedError


def load_mat_file(file_path: Path) -> ImageData:
    """
    Load MAT file format.

    Parameters
    ----------
    file_path : Path
        Path to the .mat file

    Returns
    -------
    ImageData
        Loaded data structure

    Raises
    ------
    ValueError
        If required fields are missing or invalid
    """
    # Load the MAT file
    mat_data = loadmat(file_path)

    # Migrate Scratch 2.0 fields to Scratch 3.0 format (backward compatibility)
    mat_data = migrate_scratch_2_to_3_fields(mat_data)

    # Check if 'type' field exists
    if "type" not in mat_data:
        raise ValueError("MAT file must contain 'type' field")

    # scipy.io.loadmat converts 1D arrays to 2D row vectors (1, n)
    # For profiles, we need to flatten them back to 1D
    depth_data = (
        value.flatten()
        if (value := mat_data.get("depth_data")) is not None
        and value.ndim == 2
        and value.shape[0] == 1
        else value
    )

    return ImageData(
        # scipy.io.loadmat returns strings as arrays, so we need to extract the value
        type=ImageType(_cast_mat_value(str, mat_data["type"])),
        depth_data=depth_data,
        texture_data=mat_data.get("texture_data"),
        xdim=_cast_mat_value(float, mat_data.get("xdim")),
        ydim=_cast_mat_value(float, mat_data.get("ydim")),
        crop_type=CropType(casted_value)
        if (casted_value := _cast_mat_value(str, mat_data.get("crop_type")))
        else None,
        crop_coordinates=mat_data.get("crop_coordinates"),
    )
