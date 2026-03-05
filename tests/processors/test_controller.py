from pathlib import Path

import pytest
from conversion.data_formats import Mark, ReferenceData
from conversion.plots.data_formats import ImpressionComparisonMetrics
from PIL import Image

from processors.controller import save_lr_impression_plot, save_lr_striation_plot

RESOURCES = Path(__file__).parent.parent.parent / "packages/scratch-core/tests/resources"


@pytest.fixture
def random_lr_system_path() -> Path:
    """Path to the pre-built random LR system pickle in test resources."""
    return RESOURCES / "random_lr_system.pkl"


class TestSaveLrOverviewPlot:
    """Tests for save_lr_overview_plot."""

    @pytest.mark.integration
    def test_saves_png_impression(  # noqa: PLR0913
        self,
        tmp_path: Path,
        mark_ref: Mark,
        mark_comp: Mark,
        reference_data: ReferenceData,
        impression_metrics: ImpressionComparisonMetrics,
        results_metadata: dict[str, str],
    ):
        """Output file is written and is a valid PNG."""
        output = tmp_path / "lr_plot.png"

        save_lr_impression_plot(
            reference_data=reference_data,
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            metrics=impression_metrics,
            metadata_reference={"Case": "ref-001"},
            metadata_compared={"Case": "comp-001"},
            results_metadata=results_metadata,
            score=0.5,
            lr=1.2,
            output_path=output,
        )

        assert output.exists()
        assert Image.open(output).format == "PNG"

    @pytest.mark.integration
    def test_saves_png_striation(  # noqa: PLR0913
        self,
        tmp_path: Path,
        mark_ref: Mark,
        mark_comp: Mark,
        reference_data: ReferenceData,
        results_metadata: dict[str, str],
    ):
        """Output file is written and is a valid PNG."""
        output = tmp_path / "lr_plot.png"

        save_lr_striation_plot(
            reference_data=reference_data,
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            mark_ref_aligned=mark_ref,
            mark_comp_aligned=mark_comp,
            metadata_reference={"Case": "ref-001"},
            metadata_compared={"Case": "comp-001"},
            results_metadata=results_metadata,
            score=0.5,
            lr=1.2,
            output_path=output,
        )

        assert output.exists()
        assert Image.open(output).format == "PNG"
