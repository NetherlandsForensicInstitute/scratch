from pydantic import BaseModel, Field

from computations.constants import SurfaceTerms


class PreprocessingImpressionParams(BaseModel):
    """Processing parameters for NIST preprocessing.

    :param pixel_size: Target pixel spacing in meters for resampling
    :param adjust_pixel_spacing: Adjust pixel spacing based on sample tilt
    :param surface_terms: SurfaceTerms for the leveling algorithm
    :param interp_method: Interpolation method ('nearest', 'linear', 'cubic')
    :param highpass_cutoff: High-pass filter cutoff length in meters (None to disable)
    :param lowpass_cutoff: Low-pass filter cutoff length in meters (None to disable)
    :param highpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in high pass filters.
    :param lowpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in low pass filters.
    """

    pixel_size: float | None = None
    adjust_pixel_spacing: bool = True
    surface_terms: SurfaceTerms = Field(
        default=SurfaceTerms.SPHERE,
        description="Surface fitting model for leveling operations. PLANE for planar surfaces, SPHERE for curved surfaces.",
    )
    interp_method: str = "cubic"
    highpass_cutoff: float | None = 250.0e-6
    lowpass_cutoff: float | None = 5.0e-6
    highpass_regression_order: int = 2
    lowpass_regression_order: int = 0
