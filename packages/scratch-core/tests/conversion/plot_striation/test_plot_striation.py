from pathlib import Path

import pytest
from PIL import Image

from conversion.plots.plot_striation import (
    plot_similarity,
    plot_depthmap_with_axes,
    plot_side_by_side_surfaces,
    plot_comparison_overview,
    plot_wavelength_correlation,
)
from tests.conversion.plot_striation.helper_functions import assert_valid_rgb_image

OUTPUT_DIR = Path("..")


def test_plot_similarity(profile_ref, profile_comp):
    result = plot_similarity(
        profile_ref=profile_ref,
        profile_comp=profile_comp,
        scale=1.5625e-6,
        score=0.85,
    )
    assert_valid_rgb_image(result)
    Image.fromarray(result).save(OUTPUT_DIR / "similarity_plot.png")


def test_plot_wavelength_xcorr(profile_ref, profile_comp, quality_passbands):
    result = plot_wavelength_correlation(
        profile_ref=profile_ref.flatten(),
        profile_comp=profile_comp.flatten(),
        scale=1.5625e-6,
        score=0.85,
        quality_passbands=quality_passbands,
    )
    assert_valid_rgb_image(result)
    Image.fromarray(result).save(OUTPUT_DIR / "wavelength_plot.png")


def test_plot_depthmap_with_axes(surface_ref):
    result = plot_depthmap_with_axes(
        data=surface_ref,
        scale=1.5625e-6,
        title="Test Filtered Surface",
    )
    assert_valid_rgb_image(result)
    Image.fromarray(result).save(OUTPUT_DIR / "depthmap_with_axes.png")


def test_plot_side_by_side_surfaces(surface_ref, surface_comp):
    result = plot_side_by_side_surfaces(
        data_ref=surface_ref,
        data_comp=surface_comp,
        scale=1.5625e-6,
    )
    assert_valid_rgb_image(result)
    Image.fromarray(result).save(OUTPUT_DIR / "side_by_side.png")


def test_plot_comparison_overview(
    mark_ref,
    mark_comp,
    mark_ref_aligned,
    mark_comp_aligned,
    profile_mark_ref,
    profile_mark_comp,
    metrics,
    metadata_ref,
    metadata_comp,
):
    result = plot_comparison_overview(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        profile_ref=profile_mark_ref,
        profile_comp=profile_mark_comp,
        metrics=metrics,
        metadata_ref=metadata_ref,
        metadata_comp=metadata_comp,
    )
    assert_valid_rgb_image(result)
    Image.fromarray(result).save(OUTPUT_DIR / "comparison_overview.png")


@pytest.mark.parametrize(
    "metadata_ref,metadata_comp,suffix",
    [
        (
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_001_reference",
                "Measurement ID": "striated_mark_measurement_extended",
                "Additional Info": "Some extra metadata field",
            },
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_002_comparison",
                "Measurement ID": "striated_mark_measurement_extended",
                "Additional Info": "Another extra field value",
            },
            "long_metadata",
        ),
        (
            {"ID": "A1", "Type": "ref"},
            {"ID": "B2", "Type": "comp"},
            "short_metadata",
        ),
    ],
)
def test_plot_comparison_overview_metadata_variants(
    mark_ref,
    mark_comp,
    mark_ref_aligned,
    mark_comp_aligned,
    profile_mark_ref,
    profile_mark_comp,
    metrics,
    metadata_ref,
    metadata_comp,
    suffix,
):
    result = plot_comparison_overview(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        profile_ref=profile_mark_ref,
        profile_comp=profile_mark_comp,
        metrics=metrics,
        metadata_ref=metadata_ref,
        metadata_comp=metadata_comp,
    )
    assert_valid_rgb_image(result)
    Image.fromarray(result).save(OUTPUT_DIR / f"comparison_overview_{suffix}.png")
