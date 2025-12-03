from typing import ParamSpec, Protocol

from conversion.display import clip_data, normalize
from image_generation.data_formats import LightSource, ScanImage, UnitVector3DArray

P = ParamSpec("P")


class ImageGenerator(Protocol[P]):
    def __call__(
        self,
        depth_data: ScanImage,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ScanImage: ...


def generate_3d_image(
    depth_data: ScanImage,
    *,
    light_sources: tuple[UnitVector3DArray, ...] = (
        LightSource(azimuth=90, elevation=45).unit_vector,
        LightSource(azimuth=180, elevation=45).unit_vector,
    ),
) -> ScanImage:
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
        depth_data.compute_normals(depth_data.scale_x, depth_data.scale_y)
        .apply_lights(light_sources)
        .combined.normalize()
    )


def get_array_for_display(
    depth_data: ScanImage, *, std_scaler: float = 2.0
) -> ScanImage:
    """
    Clip and normalize image data for displaying purposes.

    First the data will be clipped so that the values lie in the interval [μ - σ * S, μ + σ * S].
    Then the values are min-max normalized and scaled to the [0, 255] interval.

    :param image: An instance of `ScanImage`.
    :param std_scaler: The multiplier `S` for the standard deviation used above when clipping the image.
    :returns: An array containing the clipped and normalized image data.
    """
    clipped, lower, upper = clip_data(data=depth_data.data, std_scaler=std_scaler)
    normalized = normalize(clipped, lower, upper)
    return ScanImage(data=normalized)
