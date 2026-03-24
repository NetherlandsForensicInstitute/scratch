"""
Spatial Image Mutations
=======================

This module contains image mutations that operate on the *spatial*
properties of a `ScanImage`.

Spatial mutations:
- change the geometric arrangement of pixels
- affect spatial resolution, scale, or coordinate systems

These mutations modify *where* pixels are located or how they are
interpreted in space, without changing the meaning or intensity of
the pixel data itself.

Typical examples include:
- resampling or resizing
- cropping to a region of interest
- scaling or coordinate transforms

All mutations in this module must preserve the semantic content of
the image while adjusting its spatial representation.
"""

from typing import Self

import numpy as np
from computations.spatial import get_bounding_box
from container_models.base import BinaryMask, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox
from conversion.rotate import rotate_mask
from exceptions import ImageShapeMismatchError
from loguru import logger
from mutations.base import ImageMutation
from scipy.ndimage import binary_dilation, rotate
from skimage.transform import resize


class CropToMask(ImageMutation):
    def __init__(self, mask: BinaryMask, margin: int = 0) -> None:
        if not np.any(mask):
            raise ValueError("Can't crop to a mask where there are only 0/False")
        self.mask = mask
        self.margin = margin

    def skip_predicate(self, scan_image: ScanImage) -> bool:
        """
        Determine whether this crop should be skipped.

        Skips computation if the crop contains only ones.
        :param scan_image: Input ScanImage to resample.
        :returns: True if the crop is empty, False otherwise
        """
        if self.mask.all():
            logger.info("Skipping crop, mask is empty (containing only 1's")
            return True
        return False

    @classmethod
    def from_rotation(
        cls, rotation_angle: float, mask_before_rotation: BinaryMask
    ) -> Self:
        """
        Create a ``CropToMask`` instance for an image that will be rotated.

        The mask is first dilated and then rotated to match the rotated image.
        An additional margin is added to compensate for the dilation and
        rotation operations (margin = iterations + 2).

        :param rotation_angle: Rotation angle that will be applied to the scan image.
        :param mask_before_rotation: Mask corresponding to the image before rotation.
        :return: A ``CropToMask`` instance with the rotated mask and adjusted margin.
        """
        if np.isclose(rotation_angle, 0.0):
            return cls(mask=mask_before_rotation, margin=0)
        else:
            mask = rotate_mask(
                mask=binary_dilation(mask_before_rotation, iterations=3).astype(bool),
                rotation_angle=rotation_angle,
            )
            return cls(mask=mask, margin=5)

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Crop the image to the bounding box of the mask.
        :returns: New ScanImage cropped to the minimal bounding box containing all True mask values.
        """
        if scan_image.data.shape != self.mask.shape:
            raise ImageShapeMismatchError(
                f"image shape: {scan_image.data.shape} and crop shape: {self.mask.shape} are not equal"
            )
        y_slice, x_slice = get_bounding_box(self.mask, margin=self.margin)
        scan_image.data = scan_image.data[y_slice, x_slice]
        return scan_image


class Subsample(ImageMutation):
    """
    Subsample a `ScanImage` by skipping pixels at fixed intervals.

    This mutation reduces the spatial resolution of the image by selecting
    every N-th pixel along the X and Y axes. No interpolation is performed;
    pixel values are preserved exactly, and intermediate pixels are discarded.

    The spatial scale of the image is updated accordingly to reflect the
    increased distance between sampled pixels.

    :param step_size_x: The step size in the X-direction (columns).
    :param step_size_y: The step size in the Y-direction (rows).
    :returns: A new `ScanImage` instance with subsampled data and updated scales.
    """

    def __init__(self, step_size_x: int, step_size_y: int) -> None:
        self.step_x = step_size_x
        self.step_y = step_size_y

    def skip_predicate(self, scan_image: ScanImage) -> bool:
        if self.step_x == 1 and self.step_y == 1:
            logger.info("No subsampling needed, returning original scan image")
            return True
        return False

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        width, height = scan_image.width, scan_image.height
        if not (0 < self.step_x < width and 0 < self.step_y < height):
            raise ValueError(
                f"Step size should be positive and smaller than the image size: {(height, width)}"
            )
        logger.info(
            f"Subsampling scan image with step sizes x: {self.step_x}, y: {self.step_y}"
        )
        return ScanImage(
            data=scan_image.data[:: self.step_y, :: self.step_x].copy(),
            scale_x=scan_image.scale_x * self.step_x,
            scale_y=scan_image.scale_y * self.step_y,
        )


class Resample(ImageMutation):
    def __init__(self, target_shape: tuple[float, float]) -> None:
        """
        Constructor for initiating the Resampling.

        :param target_shape: The multipliers for the scale of the X- and Y-axis(y first).
        """
        self.target_shape_height = target_shape[0]
        self.target_shape_width = target_shape[1]

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Resample the ScanImage object using the specified resampling factors.

        :param image: Input ScanImage to resample.
        :returns: The resampled ScanImage.
        """
        anti_aliasing = (
            self.target_shape_height < scan_image.height
            and self.target_shape_width < scan_image.width
        )
        resampled_data = resize(
            image=scan_image.data,
            output_shape=(
                self.target_shape_height,
                self.target_shape_width,
            ),
            mode="edge",
            anti_aliasing=anti_aliasing,
        )
        scale_x_factor = scan_image.width / self.target_shape_width
        scale_y_factor = scan_image.height / self.target_shape_height

        logger.debug(
            f"Resampling image array to new size: {round(self.target_shape_height, 1)}/{round(self.target_shape_width, 1)} with scale: x:{round(scale_x_factor, 1)}, y:{round(scale_y_factor, 1)}"
        )

        return ScanImage(
            data=np.asarray(resampled_data, dtype=scan_image.data.dtype),
            scale_x=scan_image.scale_x * scale_x_factor,
            scale_y=scan_image.scale_y * scale_y_factor,
        )


class Rotate(ImageMutation):
    def __init__(self, rotation_angle: float):
        """Constructor to initiating the Rotate class,

        The Rotate ImageMutation is rotating the given scan_image.

        :param rotation_angle: a rotation angle in degrees positive will result in counterclockwise rotation
        """
        self.rotation_angle = rotation_angle

    def skip_predicate(self, scan_image: ScanImage) -> bool:
        """
        Determine whether this rotation should be skipped.

        Skips computation if the rotation is 0.
        :param scan_image: Input ScanImage to resample.
        :returns: True if rotation angle is 0, False otherwise
        """
        if np.isclose(self.rotation_angle, 0.0):
            logger.info(
                f"No rotation is needed, given rotation angle is close by 0, given angle : {self.rotation_angle}"
            )
            return True
        return False

    @classmethod
    def from_bounding_box(cls, bounding_box: BoundingBox) -> Self:
        """
        Calculate the rotation angle of a rectangular crop region.

        Determines the rotation angle by computing the angles between edges and the x-axis, and selecting the angle with
        the smallest absolute value.

        :param bounding_box: Bounding box of a rectangular crop region. Expects pixel coordinates,
            i.e. top-left origin, in the order [x, y].
        """
        angles = []
        for i in range(4):
            point1 = bounding_box[i]
            point2 = bounding_box[(i + 1) % 4]
            angles.append(
                np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
            )
        angle = min(angles, key=lambda x: abs(x))
        return cls(
            rotation_angle=-angle,
        )

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        scan_image.data = rotate(
            scan_image.data,
            self.rotation_angle,
            reshape=True,
            order=1,
            mode="constant",
            cval=np.nan,
        )
        return scan_image


class MakeIsotropic(ImageMutation):
    """
    Resample a `ScanImage` to isotropic resolution (equal pixel spacing in X and Y).

    If the input image already has equal scaling in both directions, the original
    instance is returned unchanged. Otherwise, the image is upsampled to the
    highest available resolution (i.e. the smallest scale value) using
    nearest-neighbor interpolation.

    This operation preserves the spatial content of the image while ensuring that
    distances in both axes are represented uniformly. NaN values are preserved and
    not interpolated.

    :returns: A new `ScanImage` instance with isotropic scaling and updated data.
    """

    @staticmethod
    def _is_isotropic(scan_image: ScanImage) -> bool:
        """Check if a scan image is isotropic within tolerance."""
        tolerance = 1e-16
        return bool(np.isclose(scan_image.scale_x, scan_image.scale_y, atol=tolerance))

    @staticmethod
    def _get_target_shape(
        scan_image: ScanImage, target_scale: float
    ) -> tuple[int, int]:
        """Get the target shape for a scan image given a target scale."""
        height, width = (
            int(round(scan_image.height * scan_image.scale_y / target_scale)),
            int(round(scan_image.width * scan_image.scale_x / target_scale)),
        )
        return height, width

    def skip_predicate(self, scan_image: ScanImage) -> bool:
        """
        Determine whether image is already isotropic and should be skipped.

        Skips computation if the image is already isotropic.

        :param scan_image: Input ScanImage to resample.
        :returns: True if image is already isotropic, False otherwise
        """
        if self._is_isotropic(scan_image):
            logger.debug(
                f"Image is already isotropic, with shape: {scan_image.data.shape}, conversion not needed."
            )
            return True
        return False

    def _upsample_image_data(
        self, scan_image: ScanImage, target_scale: float
    ) -> FloatArray2D:
        """Upsample image data in a `ScanImage` instance to a common target scale."""
        upsampled = resize(
            image=scan_image.data,
            output_shape=self._get_target_shape(scan_image, target_scale),
            mode="edge",
            anti_aliasing=False,  # Disabled for pure upsampling
            preserve_range=True,  # Keep original data intensity levels
            order=0,  # Nearest Neighbor so that NaNs appear at corresponding coordinates
        )
        return np.asarray(upsampled, dtype=np.float64)

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        target_scale = min(scan_image.scale_x, scan_image.scale_y)
        upsampled = self._upsample_image_data(scan_image, target_scale)

        return ScanImage(
            data=upsampled,
            scale_x=target_scale,
            scale_y=target_scale,
            meta_data=scan_image.meta_data,
        )
