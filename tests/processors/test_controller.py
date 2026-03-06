from pathlib import Path

import pytest
from conversion.data_formats import Mark
from conversion.likelihood_ratio import ReferenceData
from conversion.plots.data_formats import ImpressionComparisonMetrics

from processors.controller import save_lr_impression_plot, save_lr_striation_plot
from tests.processors.conftest import assert_valid_png

METADATA_REFERENCE = {"Case": "ref-001"}
METADATA_COMPARED = {"Case": "comp-001"}


class TestSaveLrOverviewPlot:
    """Tests for saving LR overview plots as PNG files."""

    @pytest.mark.integration
    def test_saves_png_impression(  # noqa: PLR0913
        self,
        tmp_path: Path,
        mark_ref: Mark,
        mark_comp: Mark,
        reference_data: ReferenceData,
        impression_metrics: ImpressionComparisonMetrics,
        results_metadata: dict[str, str],
    ) -> None:
        """Impression LR plot is written as a valid PNG."""
        output = tmp_path / "lr_plot.png"

        save_lr_impression_plot(
            reference_data=reference_data,
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            metrics=impression_metrics,
            metadata_reference=METADATA_REFERENCE,
            metadata_compared=METADATA_COMPARED,
            results_metadata=results_metadata,
            score=0.5,
            lr=1.2,
            output_path=output,
        )

        assert_valid_png(output)

    @pytest.mark.integration
    def test_saves_png_striation(  # noqa: PLR0913
        self,
        tmp_path: Path,
        mark_ref: Mark,
        mark_comp: Mark,
        reference_data: ReferenceData,
        results_metadata: dict[str, str],
    ) -> None:
        """Striation LR plot is written as a valid PNG."""
        output = tmp_path / "lr_plot.png"

        save_lr_striation_plot(
            reference_data=reference_data,
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            mark_ref_aligned=mark_ref,
            mark_comp_aligned=mark_comp,
            metadata_reference=METADATA_REFERENCE,
            metadata_compared=METADATA_COMPARED,
            results_metadata=results_metadata,
            score=0.5,
            lr=1.2,
            output_path=output,
        )

        assert_valid_png(output)
