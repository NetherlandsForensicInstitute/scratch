from typing import ParamSpec, Protocol

from image_generation.translations import clip_data, normalize
from image_generation.data_formats import LightSource, ScanImage, UnitVector3DArray

P = ParamSpec("P")


class ImageGenerator(Protocol[P]):
    def __call__(
        self,
        scan_image: ScanImage,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ScanImage: ...


def compute_3d_image(
    scan_image: ScanImage,
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

    :param scan_image: ScanImage, the data array with shape (Height, Width).
    :param light_sources: Tuple of LightSource objects defining azimuth and elevation as a unit vector. If omitted,
                         two default lights are used: (azimuth=90°, elevation=45°)
                         and (azimuth=180°, elevation=45°).

    :returns: ScanImage with the data rendered as a 3D image with the shape (Height, Width),
    """
    return (
        scan_image.compute_normals(scan_image.scale_x, scan_image.scale_y)
        .apply_lights(light_sources)
        .reduce_stack()
        .normalize()
    )


def get_array_for_display(
    scan_image: ScanImage, *, std_scaler: float = 2.0
) -> ScanImage:
    """
    Clip and normalize image data for displaying purposes.

    First the data will be clipped so that the values lie in the interval [μ - σ * S, μ + σ * S].
    Then the values are min-max normalized and scaled to the [0, 255] interval.

    :param image: An instance of `ScanImage`.
    :param std_scaler: The multiplier `S` for the standard deviation used above when clipping the image.
    :returns: ScanImage with the data rendered as a 3D image with the shape (Height, Width),
    """
    clipped, lower, upper = clip_data(data=scan_image.data, std_scaler=std_scaler)
    normalized = normalize(clipped, lower, upper)
    return ScanImage(
        data=normalized, scale_x=scan_image.scale_x, scale_y=scan_image.scale_y
    )
