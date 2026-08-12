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

#: cv2 flag per interpolation name. Consulted only by :func:`resample_nan_aware`; which one to use
#: is the caller's decision.
_INTERPOLATION_FLAGS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
}

#: A resampled pixel whose source block was covered less than this fraction by valid (non-NaN)
#: data is itself marked invalid, rather than reporting the mean of whatever little data it had.
NAN_AWARE_VALIDITY_THRESHOLD = 0.5

#: Relative tolerance for deciding that a scaling factor is already 1.0 and no resampling is needed.
#: Scales are pixel sizes in meters, on the order of 1e-6, so the comparison is deliberately made on
#: the dimensionless factor rather than on the scales themselves: 1e-6 is loose enough to absorb
#: float rounding in a scale that was computed rather than read from file, and tight enough that a
#: genuine 1-in-1e5 scale difference is still resampled. numpy's own 1e-5 default is not - it would
#: skip a 3.00003e-6 versus 3e-6 mismatch.
SCALE_MATCH_RTOL = 1e-6


def resample_scan_image_and_mask(
    scan_image: ScanImage,
    mask: BinaryMask | None = None,
    factors: tuple[float, float] | None = None,
    target_scale: float = 4e-6,
    only_downsample: bool = True,
    preserve_aspect_ratio: bool = True,
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
    :returns: Resampled ScanImage and MaskArray
    """
    if not factors:
        factors = get_scaling_factors(
            scales=(scan_image.scale_x, scan_image.scale_y), target_scale=target_scale
        )
    if only_downsample:
        factors = _clip_factors(factors, preserve_aspect_ratio)
    if np.allclose(factors, 1.0, rtol=SCALE_MATCH_RTOL, atol=0.0):
        return scan_image, mask
    image = _resample_scan_image(scan_image, factors=factors)
    if mask is not None:
        mask = resample_array_2d(mask, factors=factors)
    return image, mask


def resample_mark(mark: Mark, only_downsample: bool = False) -> Mark:
    """Resample a Mark so that the scale matches the scale specific for the mark type.

    :param mark: The Mark to resample.
    :param only_downsample: If True, only resample if it would reduce the resolution.
    :returns: The resampled Mark.
    """
    resampled_scan_image, _ = resample_scan_image_and_mask(
        mark.scan_image,
        target_scale=mark.mark_type.scale,
        only_downsample=only_downsample,
    )
    return mark.model_copy(update={"scan_image": resampled_scan_image})


def _resample_scan_image(image: ScanImage, factors: tuple[float, float]) -> ScanImage:
    """
    Resample the ScanImage object using the specified resampling factors.

    :param image: Input ScanImage to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: The resampled ScanImage.
    """
    image_array_resampled = resample_array_2d(image.data, factors=factors)
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
) -> T:
    """
    Resample a 2D array using the specified resampling factors.

    For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

    Has no notion of missing data, so a single NaN spreads over every output pixel its source
    footprint touches; :func:`resample_nan_aware` is the variant that does not.

    :param array: The array containing the image data to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: A numpy array containing the resampled image data.
    """
    factor_x, factor_y = factors
    resampled = resize(
        image=array,
        output_shape=(1 / factor_y * array.shape[0], 1 / factor_x * array.shape[1]),
        mode="edge",
        anti_aliasing=array.dtype != np.bool_ and all(factor > 1 for factor in factors),
    )
    return np.asarray(resampled, dtype=array.dtype)  # type: ignore[return-value]


def resample_nan_aware(
    array: FloatArray2D,
    factors: tuple[float, float],
    interpolation: Interpolation = "area",
) -> FloatArray2D:
    """
    Resample a 2D float array by *factors*, treating NaN as missing rather than as contagious.

    The counterpart to :func:`resample_array_2d` for callers that cannot afford to let a hole grow
    every time an image is resized; see :func:`resize_nan_aware` for how. Currently the
    surface-comparison pipeline only, which decides its own *interpolation* from the direction it is
    resampling in.

    :param array: Input 2D array, NaN where data is missing.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :param interpolation: See :data:`_INTERPOLATION_FLAGS`.
    :returns: Float64 array of the resampled image data.
    """
    factor_x, factor_y = factors
    target_shape = (
        max(1, int(round(array.shape[0] / factor_y))),
        max(1, int(round(array.shape[1] / factor_x))),
    )
    return resize_nan_aware(array, target_shape, interpolation=interpolation)


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
