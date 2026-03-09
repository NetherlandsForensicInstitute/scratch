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
from loguru import logger
from scipy.ndimage import rotate
from skimage.transform import resize

from computations.spatial import get_bounding_box
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation


class CropToMask(ImageMutation):
    def __init__(self, mask: BinaryMask) -> None:
        if not np.any(mask):
            raise ValueError("Can't crop to a mask where there are only 0/False")
        self.mask = mask

    @property
    def skip_predicate(self) -> bool:
        """
        Determine whether this crop should be skipped.

        Skips computation if the crop contains only ones.

        :returns: True if the crop is empty, False otherwise
        """
        return bool(self.mask.all())

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Crop the image to the bounding box of the mask.
        :returns: New ScanImage cropped to the minimal bounding box containing all True mask values.
        """
        if scan_image.data.shape != self.mask.shape:
            raise ImageShapeMismatchError(
                f"image shape: {scan_image.data.shape} and crop shape: {self.mask.shape} are not equal"
            )
        y_slice, x_slice = get_bounding_box(self.mask)
        scan_image.data = scan_image.data[y_slice, x_slice]
        return scan_image


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
    def __init__(self, rotation_angle: float, reverse_rotation: bool):
        """Constructor to initiating the Rotate class,

        :Note:
            if reverse_rotation is True rotation_angle will be reverse (*-1)
            if rotation_angle is more then 360, it has made a full turn and raises a ValueError
        """
        if rotation_angle >= 360 or rotation_angle <= -360:
            raise ValueError("Rotation angle must be between -359 and 359")
        self.rotation_angle = -rotation_angle if reverse_rotation else rotation_angle

    @property
    def skip_predicate(self) -> bool:
        """
        Determine whether this rotation should be skipped.

        Skips computation if the rotation is 0.

        :returns: True if rotation angle is 0, False otherwise
        """
        return True if np.isclose(self.rotation_angle, 0.0) else False

    @classmethod
    def from_bounding_box(
        cls, bounding_box: BoundingBox, reverse_rotation: bool
    ) -> Self:
        """
        Calculate the rotation angle of a rectangular crop region.

        Determines the rotation angle by computing the angles between edges and the x-axis, and selecting the angle with
        the smallest absolute value.

        :param bounding_box: Bounding box of a rectangular crop region. Expects pixel coordinates,
            i.e. top-left origin, in the order [x, y].
        :param reverse_rotation: boolean True for reversing the angle false for using the angle like normal.
        :return: The rotation angle in degrees, ranging from -180 to 180 (inclusive).
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
            rotation_angle=angle,
            reverse_rotation=reverse_rotation,
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
