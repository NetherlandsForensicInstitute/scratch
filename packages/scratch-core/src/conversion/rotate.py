import numpy as np
from scipy.ndimage import binary_dilation, rotate

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.remove_needles import remove_needles
from conversion.data_formats import CropType, CropInfo
from conversion.mask import crop_to_mask, mask_2d_array


def get_rotation_angle(
    rotation_angle: float = 0.0,
    crop_info: list[CropInfo] = None,
) -> float:
    """
    Calculate the rotation angle of a rectangular crop region.

    Determines the rotation angle either from an explicitly provided angle or by
    computing it from the corner points of a rectangular crop. When computing from
    corners, the function orders the points counter-clockwise, analyzes the angles
    of each side relative to the expected orientation of a non-rotated rectangle,
    and returns the average rotation angle.

    :param rotation_angle: Explicit rotation angle in degrees. If non-zero, this
                          value is returned directly without computation.
    :param crop_info: List of crop information objects. If provided and the first
                     crop is of type RECTANGLE, the rotation angle is computed
                     from the corner points in the crop data.
    :return: The rotation angle in degrees, ranging from -180 to 180 (exclusive
            of -180, which is normalized to 180).
    """
    if (
        rotation_angle == 0.0
        and crop_info
        and crop_info[0].crop_type == CropType.RECTANGLE
    ):
        corners = crop_info[0].data["corner"]
        # Calculate all edge lengths
        angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            angles.append((abs(angle), angle))

        # find smallest absolute angle
        rotation_angle = min(angles, key=lambda x: abs(x[0]))[1]

        # Normalize to [-90, 90] range
        if rotation_angle > 90:
            rotation_angle -= 180
        elif rotation_angle < -90:
            rotation_angle += 180

    return rotation_angle


def dilate_and_crop_image_and_mask(
    scan_image: ScanImage, mask: MaskArray, rotation_angle: float = 0.0
) -> tuple[ScanImage, MaskArray]:
    if rotation_angle != 0.0:
        dilate_steps = 3
        mask = binary_dilation(mask, iterations=dilate_steps)

    scan_image_cropped = ScanImage(
        data=crop_to_mask(scan_image.data, mask),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )
    mask_cropped = crop_to_mask(mask.astype(float), mask).astype(bool)

    return scan_image_cropped, mask_cropped


def rotate_and_crop_scan_image(
    scan_image: ScanImage, mask: MaskArray, rotation_angle: float
) -> tuple[ScanImage, MaskArray]:
    """
    Rotate and crop scan image data and mask if the rotation angle is not 0.
    # TODO: if rotation angle is multiple of 90, then no dilation

    :param scan_image: Image to rotate and crop
    :param mask: Binary mask indicating the crop region
    :param rotation_angle: Rotation angle in degrees
    :return: Tuple of (rotated_cropped_depth, cropped_mask) if rotation angle is not 0, else return scan_image
        and mask as is.
    """
    if rotation_angle != 0:
        dilate_steps = 3
        # Rotate mask (nearest-neighbor to keep binary values)
        mask_rotated = rotate(
            mask.astype(float),
            -rotation_angle,
            reshape=True,
            order=0,
            mode="constant",
            cval=0,
        )

        # Rotate depth data (linear interpolation, preserving NaNs)
        scan_image_rotated = rotate(
            scan_image.data,
            -rotation_angle,
            reshape=True,
            order=1,
            mode="constant",
            cval=np.nan,
        )

        # Find bounding box of rotated mask
        rows, cols = np.where(mask_rotated > 0.5)

        if len(rows) == 0:
            raise ValueError("Rotated mask is empty")

        # Apply margin
        margin = dilate_steps + 2
        x_min = max(0, rows.min() + margin)
        x_max = min(mask_rotated.shape[0], rows.max() - margin + 1)
        y_min = max(0, cols.min() + margin)
        y_max = min(mask_rotated.shape[1], cols.max() - margin + 1)

        # Crop to bounding box
        scan_image_cropped = scan_image_rotated[x_min:x_max, y_min:y_max]
        mask_cropped = mask_rotated[x_min:x_max, y_min:y_max]

        return ScanImage(
            data=scan_image_cropped,
            scale_x=scan_image.scale_x,
            scale_y=scan_image.scale_y,
        ), mask_cropped
    return scan_image, mask


def rotate_and_crop_scan_image_long_version(
    scan_image: ScanImage, mask: MaskArray, rotation_angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate depth data and mask to align a rectangular crop region.

    Artificially increases the mask since corner coordinates are integers
    and the rotation angle may not be 100% accurate. Handles NaN values
    in the depth data by using appropriate rotation methods.

    :param data_in_no_holes: Object containing depth_data attribute with the
                            depth image to rotate
    :param mask_crop: Binary mask indicating the crop region
    :type mask_crop: np.ndarray
    :param rotation_angle: Rotation angle in degrees (will be negated for rotation)
    :type rotation_angle: float
    :param dilate_steps: Number of dilation steps to remove from final crop
                        boundaries for smooth edges
    :type dilate_steps: int
    :return: Tuple of (rotated_cropped_data_object, cropped_mask)
    :rtype: Tuple[np.ndarray, np.ndarray]

    .. note::
       The rotation uses different methods depending on whether NaNs are present:
       - With NaNs: Uses order=1 (bilinear) with preserve_range to maintain canvas size
       - Without NaNs: Uses order=1 (bilinear) which may change canvas size
    """
    dilate_steps = 3
    scan_image_data = scan_image.data.copy()

    # Rotate the mask using nearest-neighbor interpolation
    # scipy.ndimage.rotate with order=0 is equivalent to nearest-neighbor
    mask_rotated = rotate(
        mask.astype(float),
        -rotation_angle,
        reshape=True,  # Allow canvas to grow (like '9 shears')
        order=0,  # Nearest-neighbor
        mode="constant",
        cval=0,
    )

    # Check if there are NaNs in the data
    has_nans = np.any(np.isnan(scan_image_data))

    if has_nans:
        # Calculate size difference between rotated mask and original data
        increase_x = (
            max(0, mask_rotated.shape[0] - scan_image_data.shape[0] + 2)
            if mask_rotated.shape[0] > scan_image_data.shape[0]
            else 0
        )
        increase_y = (
            max(0, mask_rotated.shape[1] - scan_image_data.shape[1] + 2)
            if mask_rotated.shape[1] > scan_image_data.shape[1]
            else 0
        )

        # Expand rotated mask if needed
        if increase_x > 0 or increase_y > 0:
            mask_crop_tmp_rot_larger = np.zeros(
                (mask_rotated.shape[0] + increase_x, mask_rotated.shape[1] + increase_y)
            )
            start_x = increase_x // 2
            start_y = increase_y // 2
            mask_crop_tmp_rot_larger[
                start_x : start_x + mask_rotated.shape[0],
                start_y : start_y + mask_rotated.shape[1],
            ] = mask_rotated
            mask_rotated = mask_crop_tmp_rot_larger

        # Create padded depth data array
        tmp_depth_data = np.full(mask_rotated.shape, np.nan)
        start_index_x = max(
            1, (tmp_depth_data.shape[0] - scan_image_data.shape[0]) // 2
        )
        start_index_y = max(
            1, (tmp_depth_data.shape[1] - scan_image_data.shape[1]) // 2
        )

        tmp_depth_data[
            start_index_x : start_index_x + scan_image_data.shape[0],
            start_index_y : start_index_y + scan_image_data.shape[1],
        ] = scan_image_data

        # Rotate with 'direct' method (preserve canvas size)
        data_in_tmp_rot = rotate(
            tmp_depth_data,
            -rotation_angle,
            reshape=False,  # Keep canvas size (like 'direct')
            order=1,  # Linear interpolation
            mode="constant",
            cval=np.nan,
        )
    else:
        # Rotate without padding (like '9 shears')
        data_in_tmp_rot = rotate(
            scan_image_data,
            -rotation_angle,
            reshape=True,  # Allow canvas to grow
            order=1,  # Linear interpolation
            mode="constant",
            cval=0,
        )

    # Create output object
    _ = type("obj", (object,), {"depth_data": data_in_tmp_rot})()

    # Determine the limits of the rotated mask
    rows, cols = np.where(mask_rotated > 0.5)  # Threshold for binary mask
    if len(rows) == 0:
        raise ValueError("Rotated mask is empty")

    x_min = rows.min()
    x_max = rows.max()
    y_min = cols.min()
    y_max = cols.max()

    # Decrease the mask and data by dilation steps for smooth boundaries
    dilate_steps = dilate_steps + 2

    # Ensure we don't go out of bounds
    x_min_crop = min(x_min + dilate_steps, data_in_tmp_rot.shape[0] - 1)
    x_max_crop = max(x_max - dilate_steps, x_min_crop + 1)
    y_min_crop = min(y_min + dilate_steps, data_in_tmp_rot.shape[1] - 1)
    y_max_crop = max(y_max - dilate_steps, y_min_crop + 1)

    # Crop the rotated data and mask
    data_out_depth = data_in_tmp_rot[
        x_min_crop : x_max_crop + 1, y_min_crop : y_max_crop + 1
    ]
    mask_crop_out = mask_rotated[
        x_min_crop : x_max_crop + 1, y_min_crop : y_max_crop + 1
    ]

    # Create output object
    data_out = type("obj", (object,), {"depth_data": data_out_depth})()

    return data_out, mask_crop_out


def rotate_crop_image_full_flow(
    scan_image: ScanImage,
    mask: MaskArray,
    rotation_angle: float = 0.0,
    crop_info: list[CropInfo] | None = None,
    times_median: float = 15,
) -> tuple[ScanImage, MaskArray]:
    rotation_angle = get_rotation_angle(rotation_angle, crop_info)
    print(f"found rotation angle: {rotation_angle}")
    scan_image_dilated, mask_dilated = dilate_and_crop_image_and_mask(
        scan_image, mask, rotation_angle
    )
    scan_image_cleaned = remove_needles(scan_image_dilated, mask_dilated, times_median)
    scan_image_masked = ScanImage(
        data=mask_2d_array(scan_image_cleaned.data, mask_dilated),
        scale_x=scan_image_cleaned.scale_x,
        scale_y=scan_image_cleaned.scale_y,
    )
    scan_image_rotated, mask_rotated = rotate_and_crop_scan_image(
        scan_image_masked, mask_dilated, rotation_angle
    )
    return scan_image_rotated, mask_rotated


def plot_rectangle(corners: list[tuple[float, float]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Close the rectangle by adding the first point at the end
    corners_closed = np.vstack([corners, corners[0]])

    plt.plot(corners_closed[:, 0], corners_closed[:, 1], "r-", linewidth=2)
    plt.plot(corners[:, 0], corners[:, 1], "bo", markersize=8)  # Show corner points
    plt.axis("equal")
    plt.grid(True)
    plt.show()
