from http import HTTPStatus
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from conversion.data_formats import Mark, MarkMetadata
from conversion.likelihood_ratio import ModelSpecs
from conversion.plots.data_formats import StriationComparisonPlots
from conversion.profile_correlator import Profile
from fastapi import HTTPException

from processors.controller import (
    compare_striation_marks,
    save_impression_comparison_plots,
    save_lr_impression_plot,
    save_lr_striation_plot,
    save_striation_comparison_plots,
)
from response_constants import ComparisonImpressionFiles, ComparisonStriationFiles

from ..helper_function import assert_valid_png, make_cell


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


class TestSaveComparisonPlots:
    """Cover save_striation_comparison_plots and save_impression_comparison_plots."""

    @staticmethod
    def _dummy_image_array():
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def test_save_striation_comparison_plots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests for saving striation score plots as PNG files."""
        img = self._dummy_image_array()
        mock_plots = Mock(spec=StriationComparisonPlots)
        mock_plots.similarity_plot = img
        mock_plots.comparison_overview = img
        mock_plots.filtered_reference_heatmap = img
        mock_plots.filtered_compared_heatmap = img
        mock_plots.side_by_side_heatmap = img

        monkeypatch.setattr(
            "processors.controller.plot_striation_comparison_results",
            lambda **kwargs: mock_plots,
        )

        save_striation_comparison_plots(
            mark_ref=Mock(spec=Mark),
            mark_comp=Mock(spec=Mark),
            mark_correlations=Mock(
                mark_reference_aligned=Mock(),
                mark_compared_aligned=Mock(),
                profile_reference_aligned=Mock(),
                profile_compared_aligned=Mock(),
                comparison_results=Mock(),
            ),
            working_dir=tmp_path,
            files_to_save=ComparisonStriationFiles,
            metadata_reference=Mock(spec=MarkMetadata),
            metadata_compared=Mock(spec=MarkMetadata),
        )

        assert_valid_png(ComparisonStriationFiles.similarity_plot.get_file_path(tmp_path))
        assert_valid_png(ComparisonStriationFiles.comparison_overview.get_file_path(tmp_path))
        assert_valid_png(ComparisonStriationFiles.filtered_reference_heatmap.get_file_path(tmp_path))
        assert_valid_png(ComparisonStriationFiles.filtered_compared_heatmap.get_file_path(tmp_path))
        assert_valid_png(ComparisonStriationFiles.side_by_side_heatmap.get_file_path(tmp_path))

    def test_save_impression_comparison_plots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests for saving impression score plots as PNG files."""
        img = self._dummy_image_array()
        mock_plots = Mock(
            comparison_overview=img,
            raw_reference_heatmap=img,
            raw_compared_heatmap=img,
            filtered_reference_heatmap=img,
            filtered_compared_heatmap=img,
            cell_reference_heatmap=img,
            cell_compared_heatmap=img,
            cell_overlay=img,
            cell_cross_correlation=img,
        )
        monkeypatch.setattr(
            "processors.controller.plot_impression_comparison_results",
            lambda **kwargs: mock_plots,
        )

        save_impression_comparison_plots(
            mark_ref=Mock(leveled_mark=Mock(), filtered_mark=Mock()),
            mark_comp=Mock(leveled_mark=Mock(), filtered_mark=Mock()),
            cmc_result=Mock(),
            comparison_params=Mock(),
            working_dir=tmp_path,
            files_to_save=ComparisonImpressionFiles,
            metadata_reference=Mock(spec=MarkMetadata),
            metadata_compared=Mock(spec=MarkMetadata),
        )

        for member in ComparisonImpressionFiles:
            assert_valid_png(member.get_file_path(tmp_path))


class TestSaveLrOverviewPlot:
    """Tests for saving LR overview plots as PNG files."""

    @pytest.mark.integration
    def test_saves_png_impression(  # noqa: PLR0913
        self,
        tmp_path: Path,
        mark_ref: Mark,
        mark_comp: Mark,
        reference_data: ModelSpecs,
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
            cells=[
                make_cell(
                    center_reference=(i * 1e-3, 0.0),
                    best_score=0.3,
                    cell_size=(1e-3, 1e-3),
                )
                for i in range(5)
            ],
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
        reference_data: ModelSpecs,
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
            score_transformed=0.5,
            reference_scores_transformed=reference_data.scores,
            lr=1.2,
            output_path=output,
        )

        assert_valid_png(output)
