from dataclasses import dataclass

from scipy.constants import micro

from conversion.leveling import SurfaceTerms


@dataclass
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
    """

    pixel_size: tuple[float, float] = (
        1.0,
        1.0,
    )  # Not set anywhere, always (1,1) or even (np.nan, np.nan)?
    adjust_pixel_spacing: bool = True  # Not set anywhere, always False? set when initialising NIST params, always True?

    # in Java Preprocessing/Impression parameter group
    level_offset: bool = True
    level_tilt: bool = True
    level_2nd: bool = True
    interp_method: str = "cubic"
    highpass_cutoff: float | None = 250.0 * micro
    lowpass_cutoff: float | None = 5.0 * micro
    regression_order_high: int = 2
    regression_order_low: int = 0
    n_contiguous = None  # Not needed?
    min_outlier_slope = None  # Not needed?
    min_pixel_area = None  # Not needed?

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
