from typing import TypeVar

import cv2
import numpy as np
from scipy.signal import resample as signal_resample

from container_models.base import BinaryMask, FloatArray1D, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark

T = TypeVar("T", FloatArray2D, BinaryMask)

#: Maps the user-facing interpolation name to the cv2 flag used when resizing. "area" is the
#: recommended choice for shrinking images (it averages every source pixel into the output rather
#: than sampling a subset of them); the others exist so different algorithms can be compared
#: empirically on real scans instead of assumed to be equivalent.
_INTERPOLATION_FLAGS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
}

#: A downsampled pixel whose source block was covered less than this fraction by valid (non-NaN)
#: data is itself marked invalid, rather than reporting the mean of whatever little data it had.
NAN_AWARE_VALIDITY_THRESHOLD = 0.5


def resample_scan_image_and_mask(
    scan_image: ScanImage,
    mask: BinaryMask | None = None,
    factors: tuple[float, float] | None = None,
    target_scale: float = 4e-6,
    only_downsample: bool = True,
    preserve_aspect_ratio: bool = True,
    interpolation: str = "area",
) -> tuple[ScanImage, BinaryMask | None]:
    """
    Resample the input image and optionally its corresponding mask.

    If `only_downsample` is True and the current resolution is already coarser than the target scale,
    no resampling is performed. If `factors` are provided, it overrides the target scale.

    :param scan_image: Input ScanImage to resample.
    :param mask: Corresponding mask array.
    :param factors: The multipliers for the scale of the X- and Y-axis. The formula used is `new_scale = factor * old_scale`.
    :param target_scale: Target scale (in meters) when `factors` are not provided.
    :param preserve_aspect_ratio: Whether to preserve the aspect ratio of the image.
    :param only_downsample: If True, only downsample data (default). If False, allow upsampling.
    :param interpolation: One of "area", "linear", "nearest", "cubic". See :data:`_INTERPOLATION_FLAGS`.
    :returns: Resampled ScanImage and MaskArray
    """
    if not factors:
        factors = get_scaling_factors(
            scales=(scan_image.scale_x, scan_image.scale_y), target_scale=target_scale
        )
    if only_downsample:
        factors = _clip_factors(factors, preserve_aspect_ratio)
    if np.allclose(factors, 1.0):
        return scan_image, mask
    image = _resample_scan_image(
        scan_image, factors=factors, interpolation=interpolation
    )
    if mask is not None:
        mask = resample_array_2d(mask, factors=factors, interpolation=interpolation)
    return image, mask


def resample_mark(
    mark: Mark, only_downsample: bool = False, interpolation: str = "area"
) -> Mark:
    """Resample a Mark so that the scale matches the scale specific for the mark type.

    :param mark: The Mark to resample.
    :param only_downsample: If True, only resample if it would reduce the resolution.
    :param interpolation: One of "area", "linear", "nearest", "cubic".
    :returns: The resampled Mark.
    """
    resampled_scan_image, _ = resample_scan_image_and_mask(
        mark.scan_image,
        target_scale=mark.mark_type.scale,
        only_downsample=only_downsample,
        interpolation=interpolation,
    )
    return mark.model_copy(update={"scan_image": resampled_scan_image})


def _resample_scan_image(
    image: ScanImage, factors: tuple[float, float], interpolation: str = "area"
) -> ScanImage:
    """
    Resample the ScanImage object using the specified resampling factors.

    :param image: Input ScanImage to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :param interpolation: One of "area", "linear", "nearest", "cubic".
    :returns: The resampled ScanImage.
    """
    image_array_resampled = resample_array_2d(
        image.data, factors=factors, interpolation=interpolation
    )
    return ScanImage(
        data=image_array_resampled,
        scale_x=image.scale_x * factors[0],
        scale_y=image.scale_y * factors[1],
    )


def resample_array_1d(
    data: FloatArray1D,
    factor: float,
) -> FloatArray1D:
    """
    Resample a 1D array with anti-aliasing.

    Uses scipy.signal.resample which applies an anti-aliasing filter before
    resampling, matching MATLAB's resample behavior.

    :param data: 1D input array.
    :param factor: Scale factor for pixel size. factor > 1 means downsampling
        (fewer output samples), factor < 1 means upsampling.
    :returns: Resampled 1D array of length max(1, round(len(data) / factor)).
    """
    n_in = len(data)
    n_out = max(1, int(round(n_in / factor)))

    if n_out == n_in:
        return data.copy()

    result: FloatArray1D = signal_resample(data, n_out)  # type: ignore[assignment]
    return result


def resample_array_2d(
    array: T,
    factors: tuple[float, float],
    interpolation: str = "area",
) -> T:
    """
    Resample a 2D array using the specified resampling factors, propagating NaN correctly.

    For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

    Boolean masks are resized directly (there is no missing-data concept for a mask). Float arrays
    are treated as potentially containing NaN: naively interpolating them would let a single NaN
    poison every output pixel it touches, so the valid and invalid pixels are averaged separately
    and recombined (see :func:`resize_nan_aware`).

    :param array: The array containing the image data to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :param interpolation: One of "area", "linear", "nearest", "cubic". "area" is recommended for
        downsampling; see :data:`_INTERPOLATION_FLAGS`.
    :returns: A numpy array containing the resampled image data.
    """
    factor_x, factor_y = factors
    new_height = max(1, int(round(array.shape[0] / factor_y)))
    new_width = max(1, int(round(array.shape[1] / factor_x)))

    if array.dtype == np.bool_:
        flag = _INTERPOLATION_FLAGS["nearest"]
        resized = cv2.resize(
            array.astype(np.float32), (new_width, new_height), interpolation=flag
        )
        return np.asarray(resized > 0.5, dtype=array.dtype)  # type: ignore[return-value]

    resized = resize_nan_aware(
        array, (new_height, new_width), interpolation=interpolation
    )
    return np.asarray(resized, dtype=array.dtype)  # type: ignore[return-value]


def resize_nan_aware(
    array: FloatArray2D,
    target_shape: tuple[int, int],
    interpolation: str = "area",
) -> FloatArray2D:
    """
    Resize a 2D float array to *target_shape*, correctly propagating missing (NaN) data.

    A plain resize would let a single NaN spread into every output pixel whose source footprint
    touches it. Instead, the valid pixels are averaged among themselves: the (fill-value-substituted)
    array and the validity mask are resized with the same interpolation and divided, so each output
    pixel is the mean of only its valid source pixels. An output pixel covered less than
    :data:`NAN_AWARE_VALIDITY_THRESHOLD` by valid source data is itself marked NaN rather than
    reporting the average of a handful of pixels.

    :param array: Input 2D array, NaN where data is missing.
    :param target_shape: ``(height, width)`` of the output.
    :param interpolation: One of "area", "linear", "nearest", "cubic".
    :returns: Float64 array of shape *target_shape*.
    """
    if interpolation not in _INTERPOLATION_FLAGS:
        raise ValueError(
            f"Unknown interpolation {interpolation!r}; choose one of {sorted(_INTERPOLATION_FLAGS)}"
        )
    flag = _INTERPOLATION_FLAGS[interpolation]
    new_height, new_width = target_shape

    valid = np.isfinite(array)
    if valid.all():
        # No missing data: a plain resize is exact and avoids the divide-by-coverage step below.
        resized = cv2.resize(
            array.astype(np.float32), (new_width, new_height), interpolation=flag
        )
        return np.asarray(resized, dtype=np.float64)

    filled = np.where(valid, array, 0.0).astype(np.float32)
    mean_of_filled = cv2.resize(filled, (new_width, new_height), interpolation=flag)
    mean_of_valid = cv2.resize(
        valid.astype(np.float32), (new_width, new_height), interpolation=flag
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        result = mean_of_filled / mean_of_valid
    result[mean_of_valid < NAN_AWARE_VALIDITY_THRESHOLD] = np.nan
    return np.asarray(result, dtype=np.float64)


def get_scaling_factors(
    scales: tuple[float, float],
    target_scale: float,
) -> tuple[float, float]:
    """
    Calculate the multipliers for a target scale.

    :param scales: Current scales (= pixel size in meters per image dimension).
    :param target_scale: Target scale (= pixel size in meters).

    :returns: The computed multipliers.
    """
    return target_scale / scales[0], target_scale / scales[1]


def _clip_factors(
    factors: tuple[float, float],
    preserve_aspect_ratio: bool,
) -> tuple[float, float]:
    """Clip the scaling factors to minimum 1.0, while keeping the aspect ratio if `preserve_aspect_ratio` is True."""
    if preserve_aspect_ratio:
        # Set the multipliers to equal values to preserve the aspect ratio
        max_factor = max(factors)
        factors = max_factor, max_factor

    return max(factors[0], 1.0), max(factors[1], 1.0)
