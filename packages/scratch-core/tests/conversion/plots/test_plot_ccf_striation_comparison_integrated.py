import numpy as np
import pytest
from matplotlib import pyplot as plt

from conversion.plots.plot_ccf_comparison_complete import plot_ccf_comparison_complete
from conversion.plots.plot_score_histograms import DensityDict

from .helper_functions import (
    create_synthetic_striation_data,
    create_sample_score_data,
    create_sample_metadata_reference,
    create_sample_metadata_compared,
    create_sample_metadata_results,
)


@pytest.fixture
def sample_metadata_reference():
    """Sample reference metadata."""
    return create_sample_metadata_reference()


@pytest.fixture
def sample_metadata_compared():
    """Sample compared metadata."""
    return create_sample_metadata_compared()


@pytest.fixture
def sample_metadata_results():
    """Sample results metadata."""
    return create_sample_metadata_results()


@pytest.fixture
def sample_surface_data():
    """Create sample surface data for heatmaps."""
    return create_synthetic_striation_data(height=100, width=50, seed=42)


@pytest.fixture
def sample_score_data():
    """Create sample score distribution data."""
    return create_sample_score_data(n_knm=1000, n_km=100, seed=42)


class TestPlotCCFComparisonComplete:
    """Test suite for plot_ccf_comparison_complete function."""

    def test_returns_figure(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_surface_data,
        sample_score_data,
    ):
        """Test that function returns a matplotlib Figure."""
        fig = plot_ccf_comparison_complete(
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=sample_surface_data,
            data_compared_filtered=sample_surface_data,
            scale_heatmap=1e-6,
            data_reference_aligned=sample_surface_data,
            data_compared_aligned=sample_surface_data,
            **sample_score_data,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_creates_expected_number_of_axes(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_surface_data,
        sample_score_data,
    ):
        """Test that the correct number of subplot axes are created."""
        fig = plot_ccf_comparison_complete(
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=sample_surface_data,
            data_compared_filtered=sample_surface_data,
            scale_heatmap=1e-6,
            data_reference_aligned=sample_surface_data,
            data_compared_aligned=sample_surface_data,
            **sample_score_data,
        )

        # Should have multiple axes for different subplots
        assert len(fig.axes) >= 8
        plt.show()
        plt.close(fig)

    @pytest.mark.parametrize(
        "new_score,score_llr_point",
        [
            (None, None),
            (0.85, None),
            (None, (0.97, 5.19)),
            (0.85, (0.97, 5.19)),
        ],
        ids=["no_markers", "with_new_score", "with_llr_point", "with_both"],
    )
    def test_optional_score_markers(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_surface_data,
        sample_score_data,
        new_score,
        score_llr_point,
    ):
        """Test with different combinations of optional score markers."""
        fig = plot_ccf_comparison_complete(
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=sample_surface_data,
            data_compared_filtered=sample_surface_data,
            scale_heatmap=1e-6,
            data_reference_aligned=sample_surface_data,
            data_compared_aligned=sample_surface_data,
            new_score=new_score,
            score_llr_point=score_llr_point,
            **sample_score_data,
        )

        assert isinstance(fig, plt.Figure)
        plt.show()
        plt.close(fig)

    def test_with_density_estimates(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_surface_data,
        sample_score_data,
    ):
        """Test with density estimates for score histograms."""
        x = np.linspace(0, 1, 50)
        densities: DensityDict = {
            "x": x,
            "km_density_at_x": 2 * x,
            "knm_density_at_x": 2 * (1 - x),
        }

        fig = plot_ccf_comparison_complete(
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=sample_surface_data,
            data_compared_filtered=sample_surface_data,
            scale_heatmap=1e-6,
            data_reference_aligned=sample_surface_data,
            data_compared_aligned=sample_surface_data,
            densities=densities,
            densities_transformed=densities,
            **sample_score_data,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestInputValidation:
    """Test input validation and error handling."""

    def test_length_mismatch_scores_labels(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_surface_data,
    ):
        """Test that mismatched scores and labels raises ValueError."""
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 90)  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch.*scores.*labels"):
            plot_ccf_comparison_complete(
                metadata_reference=sample_metadata_reference,
                metadata_compared=sample_metadata_compared,
                metadata_results=sample_metadata_results,
                data_reference_filtered=sample_surface_data,
                data_compared_filtered=sample_surface_data,
                scale_heatmap=1e-6,
                data_reference_aligned=sample_surface_data,
                data_compared_aligned=sample_surface_data,
                scores=scores,
                labels=labels,
                scores_transformed=np.random.rand(100),
                llrs=np.random.rand(50),
                llrs_at5=np.random.rand(50),
                llrs_at95=np.random.rand(50),
            )

    def test_length_mismatch_llr_arrays(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_surface_data,
    ):
        """Test that mismatched LLR arrays raise ValueError."""
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)

        with pytest.raises(ValueError, match="LLR array length mismatch"):
            plot_ccf_comparison_complete(
                metadata_reference=sample_metadata_reference,
                metadata_compared=sample_metadata_compared,
                metadata_results=sample_metadata_results,
                data_reference_filtered=sample_surface_data,
                data_compared_filtered=sample_surface_data,
                scale_heatmap=1e-6,
                data_reference_aligned=sample_surface_data,
                data_compared_aligned=sample_surface_data,
                scores=scores,
                labels=labels,
                scores_transformed=np.random.rand(100),
                llrs=np.random.rand(50),
                llrs_at5=np.random.rand(45),  # Wrong length
                llrs_at95=np.random.rand(50),
            )


@pytest.mark.parametrize(
    "metadata_reference,metadata_compared,suffix",
    [
        (
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_001_reference",
                "Measurement ID": "striated_mark_measurement_extended",
            },
            {
                "Collection": "firearms_extended_collection_name",
                "Firearm ID": "firearm_1_-_known_match_with_very_long_identifier",
                "Specimen ID": "bullet_specimen_002_comparison",
                "Measurement ID": "striated_mark_measurement_extended",
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
class TestMetadataVariants:
    """Test with different metadata lengths and content."""

    def test_handles_various_metadata_lengths(
        self,
        metadata_reference,
        metadata_compared,
        suffix,
        sample_metadata_results,
        sample_surface_data,
        sample_score_data,
    ):
        """Test that function handles both long and short metadata gracefully."""
        fig = plot_ccf_comparison_complete(
            metadata_reference=metadata_reference,
            metadata_compared=metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=sample_surface_data,
            data_compared_filtered=sample_surface_data,
            scale_heatmap=1e-6,
            data_reference_aligned=sample_surface_data,
            data_compared_aligned=sample_surface_data,
            **sample_score_data,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.integration
class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_uniform_surface_data(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_score_data,
    ):
        """Test with uniform (constant) surface data."""
        uniform_data = np.ones((100, 50)) * 1e-6

        fig = plot_ccf_comparison_complete(
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=uniform_data,
            data_compared_filtered=uniform_data,
            scale_heatmap=1e-6,
            data_reference_aligned=uniform_data,
            data_compared_aligned=uniform_data,
            **sample_score_data,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_surface_with_nan_values(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_score_data,
    ):
        """Test with NaN values in surface data."""
        data = create_synthetic_striation_data(100, 50, seed=42)
        data[10:20, 10:20] = np.nan

        fig = plot_ccf_comparison_complete(
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=data,
            data_compared_filtered=data,
            scale_heatmap=1e-6,
            data_reference_aligned=data,
            data_compared_aligned=data,
            **sample_score_data,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_surface_sizes(
        self,
        sample_metadata_reference,
        sample_metadata_compared,
        sample_metadata_results,
        sample_score_data,
    ):
        """Test with different sized surfaces."""
        data_small = create_synthetic_striation_data(50, 25, seed=42)
        data_large = create_synthetic_striation_data(200, 100, seed=43)

        fig = plot_ccf_comparison_complete(
            metadata_reference=sample_metadata_reference,
            metadata_compared=sample_metadata_compared,
            metadata_results=sample_metadata_results,
            data_reference_filtered=data_small,
            data_compared_filtered=data_large,
            scale_heatmap=1e-6,
            data_reference_aligned=data_small,
            data_compared_aligned=data_large,
            **sample_score_data,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
