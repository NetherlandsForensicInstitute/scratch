from enum import IntEnum, StrEnum, auto

from conversion.leveling import SurfaceTerms


class SurfaceOptions(StrEnum):
    """
    Surface fitting options exposed by the API.

    These options represent predefined surface leveling presets
    that map directly to internal :class:`SurfaceTerms` flag combinations.
    """

    PLANE = auto()
    SPHERE = auto()
    NONE = auto()

    def to_surface_terms(self) -> SurfaceTerms:
        """
        Convert the API surface option to the corresponding internal`SurfaceTerms` flag combination.

        :return: The corresponding internal surface term flags.
        """
        return SurfaceTerms[self.name]


class PreviewImageNormalizationBounds(IntEnum):
    low = 0
    high = 255


class SurfaceImageNormalizationBounds(IntEnum):
    low = 25
    high = 255
