from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np
from numpy.typing import NDArray
from skimage.measure import label, regionprops

type FloatArray = NDArray[np.float64]
# ymin, ymax, xmin, xmax
type BBox = tuple[int, int, int, int]


@dataclass
class ImageData:
    xdim: NDArray[np.float64]
    ydim: NDArray[np.float64]
    depth_data: NDArray[np.float64]


class Angles(NamedTuple):
    """Represents the azimuth and elevation angles of a source.

    Attributes:
        azimuth (int): Rotation around the z-axis in degrees.
        elevation (int): Elevation angle relative to the x-y plane in degrees.
    """

    azimuth: int
    elevation: int


def _mask_image(depth_data: FloatArray, mask: FloatArray) -> FloatArray:
    if depth_data.shape != mask.shape:
        raise ValueError("Mask and image data dimensions mismatch!")

    depth_data[~mask] = np.nan
    return depth_data


def _bbox_from_mask(mask: FloatArray) -> BBox | None:
    if not (props := regionprops(label(mask))):
        return None
    return props[0].bbox


def _fill_display(depth_data: FloatArray, bbox: BBox) -> FloatArray:
    ymin, ymax, xmin, xmax = bbox
    return depth_data[ymin : ymax + 1, xmin : xmax + 1]


@overload
def surface_plot(data_in): ...


@overload
def surface_plot(data_in, light_angles: tuple[Angles, ...]): ...


@overload
def surface_plot(data_in, light_angles: tuple[Angles, ...], mask: FloatArray): ...


@overload
def surface_plot(data_in, light_angles: tuple[Angles, ...], mask: FloatArray, fill_display: bool): ...


def surface_plot(
    image: ImageData,
    *,
    light_angles: tuple[Angles, ...] = (Angles(90, 45), Angles(180, 45)),
    mask: FloatArray | None = None,
    fill_display: bool = False,
):
    """Generate a surface rendering of an input image.

    Produces a 3D surface representation of ``data_in`` viewed along the
    z-axis. Two default light sources are used unless additional light angles
    are specified. Optionally, the image can be masked prior to surface
    computation, and the visible area can be adjusted to fill the display.

    Args:
        image:        Scratch data structure or image (2D array of floats).
        light_angles: Optional (n×2) tuple specifying azimuth and elevation
                      angles `[az, el]`, in degrees, for `n` light sources.
        mask:         Array of the same shape as `data_in` with foreground
                      elements > 0 and background = 0. If provided, it masks the input
                      data before surface determination.
        fill_display: Whether to stretch the masked region to fill the
                      display for better visualization when the selected region is small.

    Returns:
        Rendered surface image.

    Notes:
        - **Default light angles:**
            - (az=90, el=45): Light from the right at 45° elevation.
            - (az=180, el=45): Light from below at 45° elevation.

    """
    if mask:
        image.depth_data = _mask_image(image.depth_data, mask)
    if fill_display:
        image.depth_data = _fill_display(image.depth_data, _bbox_from_mask(mask))

    Angles(azimuth=0, elevation=90)
