from collections.abc import Iterable
from pathlib import Path

import numpy as np
from container_models.base import BinaryMask
from container_models.light_source import LightSource
from container_models.models import NormalizationBounds
from container_models.scan_image import ScanImage
from mutations.spatial import MakeIsotropic, Subsample
from numpy.typing import NDArray
from renders import (
    apply_multiple_lights,
    compute_surface_normals,
    get_scan_image_for_display,
)

from preprocessors.constants import PreviewImageNormalizationBounds, SurfaceImageNormalizationBounds
from preprocessors.exceptions import ArrayShapeMismatchError


def parse_scan_pipeline(scan_file: Path, step_size_x: int, step_size_y: int) -> ScanImage:
    """
    Parse a scan file and load it as a ScanImage.

    :param scan_file: The path to the scan file to parse.
    :param step_size_x: The number of steps to skip in the X-direction.
    :param step_size_y: The number of steps to skip in the Y-direction.
    :return: The parsed scan image data.
    """
    scan_image = Subsample(step_size_x=step_size_x, step_size_y=step_size_y)(ScanImage.from_file(scan_file))
    return MakeIsotropic()(scan_image)


def _reshape_array(array: NDArray, shape: tuple[int, int]) -> NDArray:
    if array.size != shape[0] * shape[1]:
        raise ArrayShapeMismatchError(size=array.size, target_shape=shape)

    return array.reshape(*shape)


def parse_mask_pipeline(raw_data: bytes, shape: tuple[int, int], is_bitpacked: bool) -> BinaryMask:
    """
    Convert incoming binary data to a 2D mask array.

    :param raw_data: The binary data to convert.
    :param shape: The shape of the mask array.
    :param is_bitpacked: Boolean indicating whether the binary data is bit-packed
        and should be decompressed before reshaping.
    :returns: The 2D mask array.
    """
    if not is_bitpacked:
        array = np.frombuffer(raw_data, dtype=np.bool)
        return _reshape_array(array=array, shape=shape)

    # Note: this follows our Java implementation for bitpacking
    height, width = shape
    packed = np.frombuffer(raw_data, dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little").view(np.bool)  # type: ignore
    padding = (-width) % 8
    reshaped = _reshape_array(array=unpacked, shape=(height, width + padding))
    return reshaped[:, :width]


def surface_map_pipeline(  # noqa
    parsed_scan: ScanImage,
    output_path: Path,
    light_sources: Iterable[LightSource],
    observer: LightSource,
) -> None:
    """
    Generate a 3D surface map image from scan data and save it to the specified path.

    :param parsed_scan: The scan image data to generate a surface map from.
    :param output_path: The file path where the surface map image will be saved.
    :param light_sources: Iterable of LightSource objects representing directional lights.
    :param observer: LightSource representing the observer/camera position.
    :return: The path to the saved surface map image file.
    """
    surface_scan_image = parsed_scan.model_copy()
    surface_scan_image.data = apply_multiple_lights(
        compute_surface_normals(parsed_scan),
        light_sources=light_sources,
        observer=observer,
    )
    surface_scan_image.save_as_image(
        output_path=output_path,
        normalization_bounds=NormalizationBounds(
            low=SurfaceImageNormalizationBounds.low, high=SurfaceImageNormalizationBounds.high
        ),
    )


def preview_pipeline(parsed_scan: ScanImage, output_path: Path) -> None:
    """
    Generate a preview image from scan data and save it to the specified path.

    :param parsed_scan: The scan image data to generate a preview from.
    :param output_path: The file path where the preview image will be saved.
    :return: The path to the saved preview image file.
    """
    preview_image = get_scan_image_for_display(parsed_scan)
    preview_image.save_as_image(
        output_path=output_path,
        normalization_bounds=NormalizationBounds(
            low=PreviewImageNormalizationBounds.low, high=PreviewImageNormalizationBounds.high
        ),
    )
