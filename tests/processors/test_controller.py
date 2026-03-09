from http import HTTPStatus
from pathlib import Path
from unittest.mock import Mock

import pytest
from conversion.data_formats import Mark, MarkMetadata
from conversion.likelihood_ratio import ReferenceData
from conversion.plots.data_formats import ImpressionComparisonMetrics
from conversion.profile_correlator import Profile
from fastapi import HTTPException

from processors.controller import compare_striation_marks, save_lr_impression_plot, save_lr_striation_plot
from tests.processors.conftest import assert_valid_png


class TestCompareStriationMarks:
    def test_raises_422_when_profiles_cannot_be_aligned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns 422 when marks have insufficient overlap to align."""
        monkeypatch.setattr(
            "processors.controller.correlate_striation_marks",
            lambda **kwargs: None,
        )
        dummy_mark = Mock(spec=Mark)
        dummy_profile = Mock(spec=Profile)

        with pytest.raises(HTTPException) as exc_info:
            compare_striation_marks(
                mark_ref=dummy_mark,
                mark_comp=dummy_mark,
                profile_ref=dummy_profile,
                profile_comp=dummy_profile,
            )

        assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


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
        metadata_reference: MarkMetadata,
        metadata_compared: MarkMetadata,
    ) -> None:
        """Impression LR plot is written as a valid PNG."""
        output = tmp_path / "lr_plot.png"

        save_lr_impression_plot(
            reference_data=reference_data,
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            metrics=impression_metrics,
            metadata_reference=metadata_reference,
            metadata_compared=metadata_compared,
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
        metadata_reference: MarkMetadata,
        metadata_compared: MarkMetadata,
    ) -> None:
        """Striation LR plot is written as a valid PNG."""
        output = tmp_path / "lr_plot.png"

        save_lr_striation_plot(
            reference_data=reference_data,
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            mark_ref_aligned=mark_ref,
            mark_comp_aligned=mark_comp,
            metadata_reference=metadata_reference,
            metadata_compared=metadata_compared,
            results_metadata=results_metadata,
            score=0.5,
            lr=1.2,
            output_path=output,
        )

        assert_valid_png(output)
