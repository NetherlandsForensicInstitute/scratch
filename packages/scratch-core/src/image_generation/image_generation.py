from image_generation.data_formats import ScanMap2D, LightSource, Vector3DArray


def get_surface_map(
    depth_data: ScanMap2D,
    x_dimension: float,
    y_dimension: float,
    light_angles: tuple[Vector3DArray, ...] = (
        LightSource(azimuth=90, elevation=45).vector,
        LightSource(azimuth=180, elevation=45).vector,
    ),
) -> ScanMap2D:
    """
    Compute a normalized intensity map from a 2D depth scan using directional lighting.

    This function performs the complete processing pipeline:

    1. Compute per-pixel surface normals from the depth map.
    2. Apply multiple directional light vectors to obtain per-light intensities.
    3. Combine the resulting intensity stack into a single 2D map.
    4. Normalize the combined map to a specified output range.

    :param depth_data: 2D depth map with shape (Height, Width).
    :type depth_data: IMAGE_2D_ARRAY
    :param x_dimension: Physical spacing between columns (Δx) in meters.
    :type x_dimension: float
    :param y_dimension: Physical spacing between rows (Δy) in meters.
    :type y_dimension: float
    :param light_angles: Tuple of LightSource objects defining azimuth and elevation.
                         Each light contributes a 3-element unit vector. If omitted,
                         two default lights are used: (azimuth=90°, elevation=45°)
                         and (azimuth=180°, elevation=45°).
    :type light_angles: tuple[LightSource], optional

    :returns: Normalized 2D intensity map with shape (Height, Width), suitable for
              visualization or downstream processing.
    :rtype: IMAGE_2D_ARRAY
    """
    return (
        depth_data.compute_normals(x_dimension, y_dimension)
        .apply_lights(light_angles)
        .combined.normalize()
    )
