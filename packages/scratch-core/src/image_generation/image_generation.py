from image_generation.data_formats import ScanMap2D, LightSource, UnitVector3DArray


def get_surface_map(
    depth_data: ScanMap2D,
    x_dimension: float,
    y_dimension: float,
    light_sources: tuple[UnitVector3DArray, ...] = (
        LightSource(azimuth=90, elevation=45).unit_vector,
        LightSource(azimuth=180, elevation=45).unit_vector,
    ),
) -> ScanMap2D:
    """
    Render a 3D image from 2D scan data using directional lighting.

    This function performs the complete processing pipeline:

    1. Compute per-pixel surface normals from the depth map.
    2. Apply multiple directional lights and sum the intensities to obtain the per-pixel intensities.
    3. Normalize and scale the computed pixel intensities to a specified output range.

    :param depth_data: 2D depth map with shape (Height, Width).
    :param x_dimension: Physical spacing between columns (Δx) in meters.
    :param y_dimension: Physical spacing between rows (Δy) in meters.
    :param light_sources: Tuple of LightSource objects defining azimuth and elevation as a unit vector. If omitted,
                         two default lights are used: (azimuth=90°, elevation=45°)
                         and (azimuth=180°, elevation=45°).

    :returns: Normalized 2D intensity map with shape (Height, Width), suitable for
              visualization or downstream processing.
    """
    return (
        depth_data.compute_normals(x_dimension, y_dimension)
        .apply_lights(light_sources)
        .combined.normalize()
    )
