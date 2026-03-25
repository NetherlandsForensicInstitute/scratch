from enum import IntEnum, StrEnum, auto

from conversion.leveling import SurfaceTerms

from constants import UrlFiles


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


class PrepareMarkImpressionFiles(UrlFiles):
    preview_image = "preview.png"
    surface_map_image = "surface_map.png"
    mark_data = "mark.npz"
    mark_meta = "mark.json"
    processed_data = "processed.npz"
    processed_meta = "processed.json"
    leveled_data = "leveled.npz"
    leveled_meta = "leveled.json"


class PrepareMarkStriationFiles(UrlFiles):
    preview_image = "preview.png"
    surface_map_image = "surface_map.png"
    mark_data = "mark.npz"
    mark_meta = "mark.json"
    processed_data = "processed.npz"
    processed_meta = "processed.json"
    profile_data = "profile.npz"


class GeneratedImageFiles(UrlFiles):
    preview_image = "preview.png"
    surface_map_image = "surface_map.png"


class ProcessFiles(UrlFiles):
    preview_image = "preview.png"
    surface_map_image = "surface_map.png"
    scan_image = "scan.x3p"
