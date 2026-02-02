from dataclasses import dataclass

from conversion.leveling import SurfaceTerms


@dataclass(frozen=True)
class PreprocessingImpressionParams:
    """Processing parameters for NIST preprocessing.

    :param pixel_size: Target pixel spacing in meters for resampling
    :param adjust_pixel_spacing: Adjust pixel spacing based on sample tilt
    :param level_offset: Remove constant offset
    :param level_tilt: Remove linear tilt
    :param level_2nd: Remove second-order terms
    :param interp_method: Interpolation method ('nearest', 'linear', 'cubic')
    :param highpass_cutoff: High-pass filter cutoff length in meters (None to disable)
    :param lowpass_cutoff: Low-pass filter cutoff length in meters (None to disable)
    :param highpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in high pass filters.
    :param lowpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in low pass filters.
    """

    pixel_size: float | None = None
    adjust_pixel_spacing: bool = True
    level_offset: bool = True
    level_tilt: bool = True
    level_2nd: bool = True
    interp_method: str = "cubic"
    highpass_cutoff: float | None = 250.0e-6
    lowpass_cutoff: float | None = 5.0e-6
    highpass_regression_order: int = 2
    lowpass_regression_order: int = 0

    @property
    def surface_terms(self) -> SurfaceTerms:
        """Convert leveling flags to SurfaceTerms."""
        terms = SurfaceTerms.NONE
        if self.level_offset:
            terms |= SurfaceTerms.OFFSET
        if self.level_tilt:
            terms |= SurfaceTerms.TILT_X | SurfaceTerms.TILT_Y
        if self.level_2nd:
            terms |= SurfaceTerms.ASTIG_45 | SurfaceTerms.DEFOCUS | SurfaceTerms.ASTIG_0
        return terms
