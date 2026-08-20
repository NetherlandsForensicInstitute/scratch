from dataclasses import dataclass

import numpy as np

from container_models.base import FloatArray1D


@dataclass(frozen=True)
class CMCTranslationRotation:
    """
    Immutable container for the consensus pose shared by all cells.

    :param translation: shared translation from reference to comparison image, (x, y) meters.
    :param rotation: shared rotation from reference to comparison image, degrees.
    """

    translation: tuple[float, float]
    rotation: float


@dataclass(frozen=False)
class ConsensusParameters:
    """
    Mutable container for the Procrustes fit; the rotation is filled in after the centers.

    :param rotation_center_reference: rotation_center in reference frame, (x, y) meters.
    :param rotation_center_comparison: rotation_center in comparison frame, (x, y) meters.
    :param rotation_rad: fitted rotation from reference to comparison frame, radians.
    """

    rotation_center_reference: FloatArray1D  # (2,)
    rotation_center_comparison: FloatArray1D  # (2,)
    rotation_rad: float = np.nan

    @property
    def translation(self) -> FloatArray1D:
        return self.rotation_center_comparison - self.rotation_center_reference
