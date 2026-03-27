from dataclasses import dataclass


from container_models.base import FloatArray1D


@dataclass(frozen=True)
class CMCTranslationRotation:
    """
    Mutable container for the best registration parameters found so far for one cell.
    All positional attributes are in pixel coordinates of the (rotated) comparison image.
    :param translation: shared translation from reference to comparison image, (x, y) meters.
    :param rotation: shared rotation from reference to comparison image, degrees.

    """

    translation: tuple[float, float]
    rotation: float


@dataclass(frozen=False)
class ConsensusParameters:
    """
    Immutable container for the best translation parameters
    :param rotation_center_reference: rotation_center in reference frame, (x, y) meters.
    :param rotation_center_comparison:  rotation_center in comparison frame, (x, y) meters.

    """

    rotation_center_reference: FloatArray1D  # (2,0)
    rotation_center_comparison: FloatArray1D  # (2,0)
    rotation_rad: float | None = None

    @property
    def translation(self) -> FloatArray1D:
        return self.rotation_center_comparison - self.rotation_center_reference
