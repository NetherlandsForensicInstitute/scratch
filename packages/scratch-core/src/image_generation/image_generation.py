from image_generation.data_formats import Image2DArray, LightSource, Vector3D


def get_surface_map(
    depth_data: Image2DArray,
    x_dimension: float,
    y_dimension: float,
    light_angles: tuple[Vector3D, ...] = (
        LightSource(azimuth=90, elevation=45).vector,
        LightSource(azimuth=180, elevation=45).vector,
    ),
) -> Image2DArray:
    """
    Compute a normalized intensity map from a 2D depth scan using lighting.

    This function performs the complete processing pipeline:

    1. Compute per-pixel surface normals from the depth map.
    2. Apply multiple directional light vectors to obtain per-light intensities.
    3. Combine the resulting intensity stack into a single 2D map.
    4. Normalize the combined map to a specified output range.

    Parameters
    ----------
    depth_data : IMAGE_2D_ARRAY
        2D depth map with shape ``(Height, Width)``.
    x_dimension : float
        Physical spacing between columns (Δx) in meters.
    y_dimension : float
        Physical spacing between rows (Δy) in meters.
    light_angles : tuple of LightSource, optional
        Tuple of light-source objects defining azimuth and elevation. Each light
        contributes a 3-element unit vector. If omitted, two default lights are used:
        ``(azimuth=90°, elevation=45°)`` and ``(azimuth=180°, elevation=45°)``.

    Returns
    -------
    IMAGE_2D_ARRAY
        Normalized 2D intensity map with shape ``(Height, Width)``, suitable for
        visualization or downstream processing.
    """
    return (
        depth_data.compute_normals(x_dimension, y_dimension)
        .apply_lights(light_angles)
        .combined.normalize()
    )
