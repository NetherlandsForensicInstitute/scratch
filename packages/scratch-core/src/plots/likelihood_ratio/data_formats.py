from dataclasses import dataclass

from container_models.base import FloatArray1D


@dataclass
class DensityData:
    """
    Kernel density estimates for KM and KNM score distributions.

    :param x: x values at which densities are evaluated.
    :param km_density_at_x: KM density values at x.
    :param knm_density_at_x: KNM density values at x.
    """

    x: FloatArray1D
    km_density_at_x: FloatArray1D
    knm_density_at_x: FloatArray1D


@dataclass
class HistogramData:
    """
    Input data for score histogram plots.

    :param scores: Array of score values.
    :param labels: Array of labels (0 for KNM, 1 for KM).
    :param bins: Number of bins for histogram. If None, uses 'auto' binning.
    :param densities: Optional density curves for histogram overlay.
    :param new_score: Optional score value to display as a vertical line on the histogram.
    """

    scores: FloatArray1D
    labels: FloatArray1D
    bins: int | None = None
    densities: DensityData | None = None
    new_score: float | None = None


@dataclass
class LlrTransformationData:
    """
    Input data for Log10LR transformation plots.

    :param scores: Score axis values.
    :param llrs: Log10LR values.
    :param llrs_at5: Log10LR values at 5% confidence.
    :param llrs_at95: Log10LR values at 95% confidence.
    :param score_llr_point: Optional (score, llr) coordinate to mark the score on the LLR transformation plot.
    """

    scores: FloatArray1D
    llrs: FloatArray1D
    llrs_at5: FloatArray1D
    llrs_at95: FloatArray1D
    score_llr_point: tuple[float, float] | None
