import numpy as np
from skimage.transform import rotate
from conversion.surface_comparison.models import SurfaceMap, AreaSimilarityResult
from conversion.surface_comparison.area_comparison import run_area_comparison


def test_run_area_comparison_fourier_mellin_complex():
    """Verify Fourier-Mellin handles rotation, translation, and overlap calculation."""
    # --- Arrange ---
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:150, 0:150]

    # Create a unique topography pattern
    data = (
        np.sin(x / 5)
        + np.cos(y / 5)
        + 0.5 * np.sin(x / 2) * np.cos(y / 2)
        + rng.standard_normal((150, 150)) * 0.1
    )

    spacing = np.array([2.0, 2.0])  # 2um/px
    center = np.array([0.0, 0.0])

    ref = SurfaceMap(data, spacing, center)
    ref.unfiltered_height_map = data.copy()

    # 1. Shift: dx=2 px (4um), dy=-1 px (-2um)
    # To move comp back to ref, we expect trans = [-4.0, 2.0]
    shifted_data = np.roll(data, shift=(-1, 2), axis=(0, 1))

    # 2. Rotate: 25 degrees
    expected_angle = 25.0
    comp_data = rotate(shifted_data, -expected_angle, preserve_range=True)

    comp = SurfaceMap(comp_data, spacing, center)
    comp.unfiltered_height_map = comp_data.copy()

    # --- Act ---
    # Note: If your production code requires 'params', add it here
    translation, rotation, similarity = run_area_comparison(ref, comp)

    # --- Assertions ---
    # 1. Rotation Check (F-M accuracy is usually within ~1.5 degrees)
    assert np.isclose(abs(rotation), expected_angle, atol=1.5), (
        f"Rotation {rotation} far from {expected_angle}"
    )

    # 2. Translation Check
    # Expected translation to align comp -> ref is [-4.0, 2.0]
    assert np.allclose(translation, [-4.0, 2.0], atol=1.5), (
        f"Translation {translation} incorrect"
    )

    # 3. Similarity Metrics
    assert isinstance(similarity, AreaSimilarityResult)
    assert similarity.cross_correlation_coefficient > 0.8

    # 4. Overlap Check
    # Because we rotated the image, there are black/NaN corners.
    # The overlap shouldn't be 1.0, but should be substantial (e.g., > 0.7)
    assert 0.7 < similarity.overlap_fraction < 1.0, (
        f"Overlap {similarity.overlap_fraction} unexpected"
    )


def test_area_comparison_identity_perfect_overlap():
    """Verify that identical maps yield exactly 1.0 overlap and CC."""
    rng = np.random.default_rng(123)
    data = rng.standard_normal((100, 100))

    spacing = np.array([1.0, 1.0])
    surface = SurfaceMap(data, spacing, np.array([0, 0]))
    surface.unfiltered_height_map = data.copy()

    trans, rot, sim = run_area_comparison(surface, surface)

    assert np.allclose(trans, [0, 0])
    assert np.isclose(rot, 0.0, atol=0.1)
    assert np.isclose(sim.cross_correlation_coefficient, 1.0, atol=1e-5)
    # On an identity map with no rotation, overlap must be exactly 1.0
    assert np.isclose(sim.overlap_fraction, 1.0)
