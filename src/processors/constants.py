from constants import UrlFiles


class ComparisonImpressionFiles(UrlFiles):
    comparison_overview = "comparison_overview.png"
    raw_reference_heatmap = "raw_reference_heatmap.png"
    raw_compared_heatmap = "raw_compared_heatmap.png"
    filtered_reference_heatmap = "filtered_reference_heatmap.png"
    filtered_compared_heatmap = "filtered_compared_heatmap.png"
    cell_reference_heatmap = "cell_reference_heatmap.png"
    cell_compared_heatmap = "cell_compared_heatmap.png"
    cell_overlay = "cell_overlay.png"
    cell_cross_correlation = "cell_cross_correlation.png"


class ComparisonStriationFiles(UrlFiles):
    mark_reference_aligned_surface_map = "mark_reference_aligned_surface_map.png"
    mark_compared_aligned_surface_map = "mark_compared_aligned_surface_map.png"
    filtered_reference_heatmap = "filtered_reference_heatmap.png"
    comparison_overview = "comparison_overview.png"
    mark_reference_aligned_preview = "mark_reference_aligned_preview.png"
    mark_compared_aligned_preview = "mark_compared_aligned_preview.png"
    similarity_plot = "similarity_plot.png"
    filtered_compared_heatmap = "filtered_compared_heatmap.png"
    side_by_side_heatmap = "side_by_side_heatmap.png"
    mark_compared_aligned_data = "mark_compared_aligned.npz"
    mark_compared_aligned_meta = "mark_compared_aligned.json"
    mark_reference_aligned_data = "mark_reference_aligned.npz"
    mark_reference_aligned_meta = "mark_reference_aligned.json"


class LRFiles(UrlFiles):
    lr_overview_plot = "lr_overview_plot.png"
