from enum import IntEnum

from constants import UrlFiles


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
