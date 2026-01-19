from dataclasses import dataclass


@dataclass
class PreprocessingStriationParams:
    """Processing parameters for striation preprocessing.

    :param cutoff_hi: Cutoff wavelength for shape removal in meters (highpass filter).
    :param cutoff_lo: Cutoff wavelength for noise removal in meters (lowpass filter).
    :param cut_borders_after_smoothing: If True, crop filter edge artifacts.
    :param use_mean: If True, use mean for profile extraction; if False, use median.
    :param angle_accuracy: Target angle accuracy in degrees for fine alignment.
    :param max_iter: Maximum iterations for fine alignment.
    :param extra_sub_samp: Additional subsampling factor for gradient detection.
    :param shape_noise_removal: If True, apply shape and noise removal filters.
    """

    cutoff_hi: float = 2e-3
    cutoff_lo: float = 2.5e-4
    cut_borders_after_smoothing: bool = True
    use_mean: bool = True
    angle_accuracy: float = 0.1
    max_iter: int = 25
    extra_sub_samp: int = 1
    shape_noise_removal: bool = True
