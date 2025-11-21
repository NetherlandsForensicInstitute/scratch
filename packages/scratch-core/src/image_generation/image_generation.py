from image_generation.data_formats import Image2DArray, LightAngle


def get_surface_map(
    depth_data: Image2DArray,
    x_dimension: float,
    y_dimension: float,
    light_angles: tuple[LightAngle] = (
        LightAngle(azimuth=90, elevation=45).vector,
        LightAngle(azimuth=180, elevation=45).vector,
    ),
) -> Image2DArray:
    """
    Compute a normalized surface map from a 2D depth image using lighting.

    This function performs the full pipeline:
    1. Compute per-pixel surface normals from the depth map.
    2. Apply multiple directional lights to compute diffuse and specular intensities.
    3. Combine the lighting stack into a single intensity map.
    4. Normalize the intensity map to a specified range.

    Parameters
    ----------
    depth_data : Image2DArray
        2D depth map of shape (H, W).
    x_dimension : float
        Physical spacing between columns (Δx) in meters.
    y_dimension : float
        Physical spacing between rows (Δy) in meters.
    light_angles : tuple of LightAngle.vector, optional
        Tuple of normalized 3-element vectors indicating light directions
        (default: two lights at azimuth=90°, elevation=45° and azimuth=180°, elevation=45°).

    Returns
    -------
    Image2DArray
        Normalized 2D intensity map (H, W) resulting from combined lighting,
        suitable for visualization or further processing.
    """
    return (
        depth_data.compute_normals(x_dimension, y_dimension)
        .apply_lights(light_angles)
        .combined.normalize()
    )
