import numpy as np

from container_models.scan_image import ScanImage
from conversion.data_formats import CropType, CropInfo


def rotate_image(
    scan_image: ScanImage,
    rotation_angle: float = 0.0,  # Kan deze parameter weg? is altijd 0.0
    crop_info: list[CropInfo] = None,
) -> ScanImage:
    rotated_scan_image = scan_image.model_copy(deep=True)
    if crop_info and crop_info[0].crop_type == CropType.RECTANGLE:
        points = crop_info[0]["corner"]

        # Make sure the points are ordered counter clockwise in the Java reference,
        # thus first coordinate right, second up
        mean_xy = np.mean(points, axis=0)  # coordinates center of rectangle
        a = np.degrees(
            np.arctan2(points[:, 1] - mean_xy[1], points[:, 0] - mean_xy[0])
        )  # angular position corners wrt center
        order = np.argsort(a)
        points_ordered = points[order, :]

        index = None
        for count in range(4):
            if np.array_equal(points_ordered[count, :], points[0, :]):
                index = count
                break

        if index > 0:
            points = np.vstack([points_ordered[index:, :], points_ordered[:index, :]])
        else:
            points = points_ordered

        # Convert Java coordinates to Matlab coordinates by y-flipping and adding 1
        # NOTE: First dimension is the IMAGE X-AXIS
        points[:, 1] = (scan_image.width - 1) - points[:, 1]
        points = points + 1

        P1 = points[0, :]
        P2 = points[1, :]
        P3 = points[2, :]
        P4 = points[3, :]

        # Determine the rotation angle
        dx = np.array([P2[1] - P1[1], P3[1] - P2[1], P3[1] - P4[1], P4[1] - P1[1]])
        dy = np.array([P2[0] - P1[0], P3[0] - P2[0], P3[0] - P4[0], P4[0] - P1[0]])

        # The angle is a rotation counter clockwise of rotation_angle degrees in
        # the Java reference frame. This corresponds to a rotation clockwise when
        # flipped in the y-dimension.
        rotation_angles = np.degrees(np.arctan2(dx, dy))

        # A rectangle that was not rotated should have resulted in [0, -90, 0, -90]
        # The angle over which each side of the rectangle is rotated is
        rotation_angles = rotation_angles - np.array([0, -90, 0, -90])

        # We know that all 4 rotation angles should be almost similar, but due
        # to the fact that atan2d jumps from -180 to 180 degrees, some strange results might have occurred.
        # Therefore the following checks are performed...
        while rotation_angles[1] - rotation_angles[0] > 180:
            rotation_angles[1] = rotation_angles[1] - 360
        while rotation_angles[1] - rotation_angles[0] < -180:
            rotation_angles[1] = rotation_angles[1] + 360
        while rotation_angles[2] - rotation_angles[0] > 180:
            rotation_angles[2] = rotation_angles[2] - 360
        while rotation_angles[2] - rotation_angles[0] < -180:
            rotation_angles[2] = rotation_angles[2] + 360
        while rotation_angles[3] - rotation_angles[0] > 180:
            rotation_angles[3] = rotation_angles[3] - 360
        while rotation_angles[3] - rotation_angles[0] < -180:
            rotation_angles[3] = rotation_angles[3] + 360

        # Calc the average rotation
        rotation_angle = np.mean(rotation_angles)

        if rotation_angle == -180:
            rotation_angle = 180

    return rotated_scan_image


def simpler_rotate(
    scan_image: ScanImage,
    rotation_angle: float = 0.0,  # Kan deze parameter weg? is altijd 0.0
    crop_info: list[CropInfo] = None,
) -> ScanImage:
    rotated_scan_image = scan_image.model_copy(deep=True)
    if crop_info and crop_info[0].crop_type == CropType.RECTANGLE:
        # Get rectangle corner points
        point_array = np.array(crop_info[0][1]["corner"], dtype=float)

        # Sort points counter-clockwise starting from the first original point
        mean_xy = np.mean(point_array, axis=0)
        angles = np.degrees(
            np.arctan2(point_array[:, 1] - mean_xy[1], point_array[:, 0] - mean_xy[0])
        )
        order = np.argsort(angles)
        point_array_sorted = point_array[order, :]

        # Find where the first original point ended up and rotate array to start there
        first_point_idx = np.where((point_array_sorted == point_array[0]).all(axis=1))[
            0
        ][0]
        point_array = np.roll(point_array_sorted, -first_point_idx, axis=0)

        # Convert Java coordinates to Matlab coordinates by y-flipping and adding 1
        point_array[:, 1] = (scan_image.width - 1) - point_array[:, 1]
        point_array += 1

        # Calculate rotation angle from consecutive points
        P = point_array
        dx = np.diff(P[:, 1], append=P[0, 1])  # Wrap around to first point
        dy = np.diff(P[:, 0], append=P[0, 0])
        rotation_angles = np.degrees(np.arctan2(dx, dy)) - np.array([0, -90, 0, -90])

        # Unwrap angles to handle -180/180 discontinuity
        rotation_angles = np.unwrap(rotation_angles, period=360)

        # Calculate average rotation
        rotation_angle = np.mean(rotation_angles)
        if rotation_angle == -180:
            rotation_angle = 180
    return rotated_scan_image
