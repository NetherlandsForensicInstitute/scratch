from typing import Literal, TypeVar

import cv2
import numpy as np
from scipy.signal import resample as signal_resample
from skimage.transform import resize

from container_models.base import BinaryMask, FloatArray1D, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark

T = TypeVar("T", FloatArray2D, BinaryMask)

Interpolation = Literal["area", "linear", "nearest", "cubic"]
#: How a 2D array is shrunk. ``nan_aware`` averages only the valid pixels of each source block, so
#: missing data does not spread; ``legacy`` is the original ``skimage`` resize, which propagates a
#: NaN into every output pixel it touches. Both are kept so they can be compared on real scans.
ResampleMethod = Literal["nan_aware", "legacy"]

#: cv2 flag per interpolation name. "area" is recommended for shrinking (it averages every source
#: pixel rather than sampling a subset); the rest exist to compare algorithms on real scans.
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
    interpolation: Interpolation = "area",
    method: ResampleMethod = "nan_aware",
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
    :param interpolation: See :data:`_INTERPOLATION_FLAGS`.
    :param method: See :data:`ResampleMethod`.
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
        scan_image, factors=factors, interpolation=interpolation, method=method
    )
    if mask is not None:
        mask = resample_array_2d(
            mask, factors=factors, interpolation=interpolation, method=method
        )
    return image, mask


def resample_mark(
    mark: Mark,
    only_downsample: bool = False,
    interpolation: Interpolation = "area",
    method: ResampleMethod = "nan_aware",
) -> Mark:
    """Resample a Mark so that the scale matches the scale specific for the mark type.

    :param mark: The Mark to resample.
    :param only_downsample: If True, only resample if it would reduce the resolution.
    :param interpolation: See :data:`_INTERPOLATION_FLAGS`.
    :param method: See :data:`ResampleMethod`.
    :returns: The resampled Mark.
    """
    resampled_scan_image, _ = resample_scan_image_and_mask(
        mark.scan_image,
        target_scale=mark.mark_type.scale,
        only_downsample=only_downsample,
        interpolation=interpolation,
        method=method,
    )
    return mark.model_copy(update={"scan_image": resampled_scan_image})


def _resample_scan_image(
    image: ScanImage,
    factors: tuple[float, float],
    interpolation: Interpolation = "area",
    method: ResampleMethod = "nan_aware",
) -> ScanImage:
    """
    Resample the ScanImage object using the specified resampling factors.

    :param image: Input ScanImage to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :param interpolation: See :data:`_INTERPOLATION_FLAGS`.
    :param method: See :data:`ResampleMethod`.
    :returns: The resampled ScanImage.
    """
    image_array_resampled = resample_array_2d(
        image.data, factors=factors, interpolation=interpolation, method=method
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
    interpolation: Interpolation = "area",
    method: ResampleMethod = "nan_aware",
) -> T:
    """
    Resample a 2D array using the specified resampling factors.

    For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

    Boolean masks are resized nearest-neighbor, since a mask has no missing-data concept. Float
    arrays are resized by *method*.

    :param array: The array containing the image data to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :param interpolation: See :data:`_INTERPOLATION_FLAGS`. Ignored by the ``legacy`` method.
    :param method: See :data:`ResampleMethod`.
    :returns: A numpy array containing the resampled image data.
    """
    factor_x, factor_y = factors
    target_shape = (
        max(1, int(round(array.shape[0] / factor_y))),
        max(1, int(round(array.shape[1] / factor_x))),
    )

    if method == "legacy":
        resized = resize_legacy(array, target_shape, factors=factors)
    elif array.dtype == np.bool_:
        resized = _resize(array.astype(np.float32), target_shape, "nearest") > 0.5
    else:
        resized = resize_nan_aware(array, target_shape, interpolation=interpolation)  # type: ignore[arg-type]
    return np.asarray(resized, dtype=array.dtype)  # type: ignore[return-value]


def resize_nan_aware(
    array: FloatArray2D,
    target_shape: tuple[int, int],
    interpolation: Interpolation = "area",
) -> FloatArray2D:
    """
    Resize a 2D float array to *target_shape*, correctly propagating missing (NaN) data.

    A plain resize would let a single NaN spread into every output pixel whose source footprint
    touches it. Instead the valid pixels are averaged among themselves: the zero-substituted array
    and the validity mask are resized identically and divided, so each output pixel is the mean of
    only its valid sources. An output pixel covered less than
    :data:`NAN_AWARE_VALIDITY_THRESHOLD` by valid data is itself marked NaN.

    :param array: Input 2D array, NaN where data is missing.
    :param target_shape: ``(height, width)`` of the output.
    :param interpolation: See :data:`_INTERPOLATION_FLAGS`.
    :returns: Float64 array of shape *target_shape*.
    """
    valid = np.isfinite(array)
    if valid.all():
        # No missing data: a plain resize is exact and avoids the divide-by-coverage step below.
        return np.asarray(_resize(array, target_shape, interpolation), dtype=np.float64)

    mean_of_filled = _resize(np.where(valid, array, 0.0), target_shape, interpolation)
    mean_of_valid = _resize(valid, target_shape, interpolation)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = mean_of_filled / mean_of_valid
    result[mean_of_valid < NAN_AWARE_VALIDITY_THRESHOLD] = np.nan
    return np.asarray(result, dtype=np.float64)


def resize_legacy(
    array: FloatArray2D | BinaryMask,
    target_shape: tuple[int, int],
    factors: tuple[float, float],
) -> FloatArray2D:
    """
    Resize a 2D array the way this package did before :func:`resize_nan_aware` existed.

    ``skimage.transform.resize`` with edge padding, anti-aliased when every axis is being shrunk.
    Kept as a comparison baseline: unlike :func:`resize_nan_aware` it has no notion of missing data,
    so a single NaN spreads over every output pixel its source footprint touches.

    :param array: Input 2D array.
    :param target_shape: ``(height, width)`` of the output.
    :param factors: Scale multipliers, used only to decide whether to anti-alias.
    :returns: Resized array.
    """
    return resize(
        image=array,
        output_shape=target_shape,
        mode="edge",
        anti_aliasing=array.dtype != np.bool_ and all(factor > 1 for factor in factors),
    )


def _resize(
    array: FloatArray2D | BinaryMask,
    target_shape: tuple[int, int],
    interpolation: Interpolation,
) -> FloatArray2D:
    """Resize with cv2, which needs float32 input and ``(width, height)`` output order."""
    if interpolation not in _INTERPOLATION_FLAGS:
        raise ValueError(
            f"Unknown interpolation {interpolation!r}; choose one of {sorted(_INTERPOLATION_FLAGS)}"
        )
    height, width = target_shape
    return np.asarray(
        cv2.resize(
            array.astype(np.float32),
            (width, height),
            interpolation=_INTERPOLATION_FLAGS[interpolation],
        ),
        dtype=np.float32,
    )


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
