"""Utility functions for file I/O operations."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def check_if_file_exists(file_path: Path) -> None:
    """
    Check if a file exists at the given path.

    :param file_path: Path to check
    :raises FileNotFoundError: If file does not exist
    """
    if not file_path.is_file():
        raise FileNotFoundError(f'File "{file_path}" does not exist')


def load_json(file_path: Path) -> dict[str, Any]:
    """
    Load and parse JSON data from a file.

    :param file_path: Path to JSON file
    :returns: Dictionary containing parsed JSON data
    :raises FileNotFoundError: If file does not exist
    :raises json.JSONDecodeError: If file contains invalid JSON
    """
    with file_path.open("r") as f:
        return json.load(f)


def load_compressed_binary(file_path: Path) -> NDArray:
    """
    Load numpy array from compressed NPZ file.

    :param file_path: Path to NPZ file
    :returns: Numpy array containing the loaded data
    :raises FileNotFoundError: If file does not exist
    :raises KeyError: If 'data' key not found in NPZ file
    """
    with np.load(file_path) as zipped:
        return zipped["data"]


def save_as_json(data: str, file_path: Path) -> None:
    """
    Save string data as JSON file.

    :param data: JSON string to save
    :param file_path: Base file path
    """
    json_path = file_path.with_suffix(".json")
    json_path.write_text(data)


def save_as_compressed_binary(array: NDArray, file_path: Path) -> None:
    """
    Save numpy array as a compressed NPZ file.

    :param array: Numpy array to save
    :param file_path: Base file path
    """
    npz_path = file_path.with_suffix(".npz")
    np.savez_compressed(npz_path, data=array)
