
from enum import StrEnum
from pathlib import Path


class UrlFiles(StrEnum):
    def get_file_path(self, working_dir: Path) -> Path:
        """Return path to the file with the given working directory."""
        return working_dir / self.value

    def generate_url(self, access_url: str) -> str:
        """Generate the url to retrieve the file via the endpoint."""
        return f"{access_url}/{self.value}"


class ComparisonImpressionFiles(UrlFiles):
    mark_ref_surfacemap = "mark_ref_surfacemap.png"
    mark_comp_surfacemap = "mark_comp_surfacemap.png"
    filtered_reference_heatmap = "filtered_reference_heatmap.png"
    comparison_overview = "comparison_overview.png"
    mark_ref_filtered_moved_surfacemap = "mark_ref_filtered_moved_surfacemap.png"
    mark_ref_filtered_bb_surfacemap = "mark_ref_filtered_bb_surfacemap.png"
    mark_comp_filtered_bb_surfacemap = "mark_comp_filtered_bb_surfacemap.png"
    mark_comp_filtered_all_bb_surfacemap = "mark_comp_filtered_all_bb_surfacemap.png"
    cell_accf_distribution = "cell_accf_distribution.png"


class ComparisonStriationFiles(UrlFiles):
    mark_ref_surfacemap = "mark_ref_surfacemap.png"
    mark_comp_surfacemap = "mark_comp_surfacemap.png"
    filtered_reference_heatmap = "filtered_reference_heatmap.png"
    comparison_overview = "comparison_overview.png"
    mark_ref_preview = "mark_ref_preview.png"
    mark_comp_preview = "mark_comp_preview.png"
    similarity_plot = "similarity_plot.png"
    filtered_compared_heatmap = "filtered_compared_heatmap.png"
    side_by_side_heatmap = "side_by_side_heatmap.png"


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


class LRFiles(UrlFiles):
    lr_overview_plot = "lr_overview_plot.png"
