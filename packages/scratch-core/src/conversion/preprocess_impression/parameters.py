from pydantic import BaseModel, Field
from conversion.data_formats import SurfaceTerms, SurfaceTermsAnnotated


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

    pixel_size: float | None = Field(default=None)
    adjust_pixel_spacing: bool = Field(default=True)
    surface_terms: SurfaceTermsAnnotated = Field(
        default=SurfaceTerms.SPHERE,
        description=(
            "Surface fitting model for leveling operations. PLANE for planar surfaces, SPHERE for curved surfaces. "
            "Accepts string (e.g. 'plane', 'PLANE') or int (e.g. 1) values."
        ),
        examples=["plane", "sphere"],
    )
    interp_method: str = Field(default="cubic")
    highpass_cutoff: float | None = Field(default=250.0e-6)
    lowpass_cutoff: float | None = Field(default=5.0e-6)
    highpass_regression_order: int = Field(default=2)
    lowpass_regression_order: int = Field(default=0)
