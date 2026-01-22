"""
Tests for plot_results_profile_nfi module.

Run with: pytest test_plot_results_profile_nfi.py
"""

import numpy as np
from pathlib import Path

from PIL import Image

from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType, CropType
from conversion.plot_striation import (
    plot_similarity,
    WavelengthXcorrData,
    plot_wavelength_xcorr,
    plot_depthmap_with_axes,
    plot_side_by_side_surfaces,
    CorrelationMetrics,
    plot_comparison_overview,
)

OUTPUT_DIR = Path(".")


def create_synthetic_surface_data(
    height: int = 256,
    width: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Create synthetic striation surface data with horizontal grooves.

    :param height: Image height in pixels.
    :param width: Image width in pixels.
    :param seed: Random seed for reproducibility.
    :returns: Surface data in meters.
    """
    rng = np.random.default_rng(seed)

    y = np.arange(height)
    x = np.arange(width)
    X, Y = np.meshgrid(x, y)

    # Multiple frequency striations (horizontal bands)
    surface = np.zeros((height, width))
    surface += 0.5 * np.sin(2 * np.pi * Y / 100)
    surface += 0.3 * np.sin(2 * np.pi * Y / 25)
    surface += 0.1 * np.sin(2 * np.pi * Y / 8)
    surface += 0.2 * np.sin(2 * np.pi * X / 150)
    surface += 0.1 * rng.standard_normal((height, width))

    # Convert to meters (typical range -5 to 5 µm)
    return surface * 1e-6


def create_synthetic_profile_data(
    length: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Create synthetic 1D profile data.

    :param length: Profile length in pixels.
    :param seed: Random seed.
    :returns: Profile data in meters (1, N) array.
    """
    rng = np.random.default_rng(seed)

    x = np.arange(length)
    profile = np.zeros(length)
    profile += 0.5 * np.sin(2 * np.pi * x / 80)
    profile += 0.3 * np.sin(2 * np.pi * x / 20)
    profile += 0.1 * np.sin(2 * np.pi * x / 6)
    profile += 0.05 * rng.standard_normal(length)

    # Convert to meters, return as (1, N) array
    return (profile * 1e-6).reshape(1, -1)


def create_mock_mark(
    height: int = 256,
    width: int = 200,
    scale: float = 1.5625e-6,
    seed: int = 42,
) -> Mark:
    """Create a MockMark with synthetic surface data."""
    data = create_synthetic_surface_data(height, width, seed)
    return Mark(
        scan_image=ScanImage(data=data, scale_x=scale, scale_y=scale),
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
    )


def create_mock_profile_mark(
    length: int = 200,
    scale: float = 1.5625e-6,
    seed: int = 42,
) -> Mark:
    """Create a MockMark with synthetic profile data."""
    data = create_synthetic_profile_data(length, seed)
    return Mark(
        scan_image=ScanImage(data=data, scale_x=scale, scale_y=scale),
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
    )


def test_plot_similarity():
    """Test the similarity plot function."""
    profile_ref = create_synthetic_profile_data(length=200, seed=42)
    profile_comp = create_synthetic_profile_data(length=200, seed=43)
    scale = 1.5625e-6

    result = plot_similarity(
        profile_ref=profile_ref,
        profile_comp=profile_comp,
        scale=scale,
        score=0.85,
    )

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    Image.fromarray(result).save(OUTPUT_DIR / "similarity_plot.png")


def test_plot_wavelength_xcorr():
    """Test the wavelength cross-correlation plot function."""
    profile_ref = create_synthetic_profile_data(length=200, seed=42)
    profile_comp = create_synthetic_profile_data(length=200, seed=43)
    scale = 1.5625e-6

    xcorr_data = WavelengthXcorrData(
        xcorr_full_range=0.85,
        quality_passbands=np.array(
            [
                [5, 250],
                [100, 250],
                [50, 100],
                [25, 50],
                [10, 25],
                [5, 10],
            ]
        ),
        range_dependent_xcorrs=np.array([0.85, 0.78, 0.65, 0.45, 0.30, 0.15]),
    )

    result = plot_wavelength_xcorr(
        profile_ref=profile_ref,
        profile_comp=profile_comp,
        scale=scale,
        xcorr_data=xcorr_data,
    )

    assert result.image.ndim == 3, f"Expected 3D array, got {result.image.ndim}D"
    assert result.image.shape[2] == 3, (
        f"Expected RGB, got {result.image.shape[2]} channels"
    )
    assert result.data == xcorr_data, "Data should be passed through"

    Image.fromarray(result.image).save(OUTPUT_DIR / "wavelength_plot.png")


def test_plot_depthmap_with_axes():
    """Test the depthmap plot function."""
    data = create_synthetic_surface_data(height=256, width=200, seed=42)
    scale = 1.5625e-6

    result = plot_depthmap_with_axes(
        data=data,
        scale=scale,
        title="Test Filtered Surface",
    )

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"

    Image.fromarray(result).save(OUTPUT_DIR / "depthmap_with_axes.png")


def test_plot_side_by_side_surfaces():
    """Test the side-by-side surfaces plot function."""
    data_ref = create_synthetic_surface_data(height=200, width=200, seed=42)
    data_comp = create_synthetic_surface_data(height=200, width=220, seed=43)
    scale = 1.5625e-6

    result = plot_side_by_side_surfaces(
        data_ref=data_ref,
        data_comp=data_comp,
        scale=scale,
    )

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"

    Image.fromarray(result).save(OUTPUT_DIR / "side_by_side.png")


def test_plot_comparison_overview():
    """Test the comparison overview plot function."""
    mark_ref = create_mock_mark(height=256, width=200, seed=42)
    mark_comp = create_mock_mark(height=256, width=220, seed=43)
    mark_ref_aligned = create_mock_mark(height=200, width=200, seed=44)
    mark_comp_aligned = create_mock_mark(height=200, width=200, seed=45)
    profile_ref = create_mock_profile_mark(length=200, seed=46)
    profile_comp = create_mock_profile_mark(length=200, seed=47)

    metrics = CorrelationMetrics(
        score=0.85,
        shift=12.5,
        overlap=80.4,
        sq_a=0.2395,
        sq_b=0.7121,
        sq_b_minus_a=0.6138,
        sq_ratio=297.3765,
        sign_diff_dsab=220.94,
        data_spacing=1.5625,
        cutoff_low_pass=5,
        cutoff_high_pass=250,
        date_report="2025-12-02",
        mark_type="Aperture shear striation",
    )

    metadata_ref = {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_1",
        "Measurement ID": "striated_mark",
    }

    metadata_comp = {
        "Collection": "firearms",
        "Firearm ID": "firearm_1_-_known_match",
        "Specimen ID": "bullet_2",
        "Measurement ID": "striated_mark",
    }

    result = plot_comparison_overview(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        profile_ref=profile_ref,
        profile_comp=profile_comp,
        metrics=metrics,
        metadata_ref=metadata_ref,
        metadata_comp=metadata_comp,
    )

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"

    Image.fromarray(result).save(OUTPUT_DIR / "comparison_overview.png")


def test_plot_comparison_overview_long_metadata():
    """Test the comparison overview with longer keys and values."""
    mark_ref = create_mock_mark(height=256, width=200, seed=42)
    mark_comp = create_mock_mark(height=256, width=220, seed=43)
    mark_ref_aligned = create_mock_mark(height=200, width=200, seed=44)
    mark_comp_aligned = create_mock_mark(height=200, width=200, seed=45)
    profile_ref = create_mock_profile_mark(length=200, seed=46)
    profile_comp = create_mock_profile_mark(length=200, seed=47)

    metrics = CorrelationMetrics(
        score=0.85,
        shift=12.5,
        overlap=80.4,
        sq_a=0.2395,
        sq_b=0.7121,
        sq_b_minus_a=0.6138,
        sq_ratio=297.3765,
        sign_diff_dsab=220.94,
        data_spacing=1.5625,
        cutoff_low_pass=5,
        cutoff_high_pass=250,
        date_report="2025-12-02",
        mark_type="Aperture shear striation with extended description",
    )

    # Test with longer keys and values
    metadata_ref = {
        "Collection": "firearms_extended_collection_name",
        "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
        "Specimen ID": "bullet_specimen_001_reference",
        "Measurement ID": "striated_mark_measurement_extended",
        "Additional Info": "Some extra metadata field",
    }

    metadata_comp = {
        "Collection": "firearms_extended_collection_name",
        "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
        "Specimen ID": "bullet_specimen_002_comparison",
        "Measurement ID": "striated_mark_measurement_extended",
        "Additional Info": "Another extra field value",
    }

    result = plot_comparison_overview(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        profile_ref=profile_ref,
        profile_comp=profile_comp,
        metrics=metrics,
        metadata_ref=metadata_ref,
        metadata_comp=metadata_comp,
    )

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"

    Image.fromarray(result).save(OUTPUT_DIR / "comparison_overview_long_metadata.png")


def test_plot_comparison_overview_short_metadata():
    """Test the comparison overview with short keys and values."""
    mark_ref = create_mock_mark(height=256, width=200, seed=42)
    mark_comp = create_mock_mark(height=256, width=220, seed=43)
    mark_ref_aligned = create_mock_mark(height=200, width=200, seed=44)
    mark_comp_aligned = create_mock_mark(height=200, width=200, seed=45)
    profile_ref = create_mock_profile_mark(length=200, seed=46)
    profile_comp = create_mock_profile_mark(length=200, seed=47)

    metrics = CorrelationMetrics(
        score=0.85,
        shift=12.5,
        overlap=80.4,
    )

    # Test with short keys and values
    metadata_ref = {
        "ID": "A1",
        "Type": "ref",
    }

    metadata_comp = {
        "ID": "B2",
        "Type": "comp",
    }

    result = plot_comparison_overview(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        profile_ref=profile_ref,
        profile_comp=profile_comp,
        metrics=metrics,
        metadata_ref=metadata_ref,
        metadata_comp=metadata_comp,
    )

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[2] == 3, f"Expected RGB, got {result.shape[2]} channels"

    Image.fromarray(result).save(OUTPUT_DIR / "comparison_overview_short_metadata.png")
