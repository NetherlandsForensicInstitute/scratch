from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingStriationParams:
    """
    Processing parameters for striation preprocessing.

    :param highpass_cutoff: Cutoff wavelength for shape removal in meters (highpass filter).
    :param lowpass_cutoff: Cutoff wavelength for noise removal in meters (lowpass filter).
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts.
    :param use_mean: If True, use mean for profile extraction; if False, use median.
    :param angle_accuracy: Target angle accuracy in degrees for fine alignment.
    :param max_iter: Maximum iterations for fine alignment.
    :param subsampling_factor: Subsampling factor for gradient detection (higher = faster but less precise).
    """

    highpass_cutoff: float = 2e-3
    lowpass_cutoff: float = 2.5e-4
    cut_borders_after_smoothing: bool = True
    use_mean: bool = True
    angle_accuracy: float = 0.1
    max_iter: int = 25
    subsampling_factor: int = 1
