from pathlib import Path

import pytest
from conversion.surface_comparison.models import Cell
from pydantic import ValidationError

from processors.schemas import (
    CalculateLR,
    CalculateLRImpression,
    CalculateLRStriation,
    MarkDirectories,
)


class TestMarkDirectories:
    """Tests for the MarkDirectories schema."""

    def test_should_accept_valid_directories(self, mark_dirs: tuple[Path, Path]) -> None:
        """Valid existing directories are accepted."""
        mark_dir_ref, mark_dir_comp = mark_dirs
        schema = MarkDirectories(mark_dir_ref=mark_dir_ref, mark_dir_comp=mark_dir_comp)
        assert schema.mark_dir_ref == mark_dir_ref
        assert schema.mark_dir_comp == mark_dir_comp

    @pytest.mark.parametrize("field", ["mark_dir_ref", "mark_dir_comp"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_dirs: tuple[Path, Path]
    ) -> None:
        """Non-existent paths and file paths raise ValidationError."""
        mark_dir_ref, mark_dir_comp = mark_dirs
        fields = {"mark_dir_ref": mark_dir_ref, "mark_dir_comp": mark_dir_comp} | {field: invalid_path}
        with pytest.raises(ValidationError):
            MarkDirectories.model_validate(fields)


class TestCalculateLR:
    """Tests for the CalculateLR schema."""

    def test_should_accept_valid_input(
        self, mark_dirs: tuple[Path, Path], lr_system_path: Path, base_kwargs: dict
    ) -> None:
        """Valid directories and lr_system_path file are accepted."""
        mark_dir_ref, mark_dir_comp = mark_dirs
        schema = CalculateLR(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            lr_system_path=lr_system_path,
            **base_kwargs,
        )
        assert schema.lr_system_path == lr_system_path

    @pytest.mark.parametrize(
        "invalid_lr_system",
        [
            pytest.param(Path() / "nonexistent", id="nonexistent"),
            pytest.param(Path(__file__), id="file"),
        ],
    )
    def test_should_reject_invalid_lr_system_path(
        self, mark_dirs: tuple[Path, Path], invalid_lr_system: Path, base_kwargs: dict
    ) -> None:
        """Non-existent path and file path for lr_system_path raise ValidationError."""
        mark_dir_ref, mark_dir_comp = mark_dirs
        with pytest.raises(ValidationError):
            CalculateLR(
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                lr_system_path=invalid_lr_system,
                **base_kwargs,
            )

    @pytest.mark.parametrize("field", ["mark_dir_ref", "mark_dir_comp", "lr_system_path"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_dirs: tuple[Path, Path], lr_system_path: Path
    ) -> None:
        """Non-existent paths and file paths for directory fields raise ValidationError."""
        mark_dir_ref, mark_dir_comp = mark_dirs
        fields = {
            "mark_dir_ref": mark_dir_ref,
            "mark_dir_comp": mark_dir_comp,
            "lr_system_path": lr_system_path,
        } | {field: invalid_path}
        with pytest.raises(ValidationError):
            CalculateLR.model_validate(fields)


class TestCalculateLRImpression:
    """Tests for the CalculateLRImpression schema."""

    def test_should_accept_valid_input(self, impression_kwargs: dict) -> None:
        """Valid input including score, n_cells, and impression LR parameters is accepted."""
        schema = CalculateLRImpression(**impression_kwargs)
        assert isinstance(schema.cells[0], Cell)

    @pytest.mark.parametrize("score", [0, 1, 100])
    def test_should_accept_score_within_n_cells(self, impression_kwargs: dict, score: int) -> None:
        """Score accepts any non-negative integer up to n_cells."""
        schema = CalculateLRImpression(**impression_kwargs | {"score": score, "n_cells": 100})
        assert schema.score == score

    def test_should_accept_score_equal_to_n_cells(self, impression_kwargs: dict) -> None:
        """Score equal to n_cells is valid (all cells match)."""
        schema = CalculateLRImpression(**impression_kwargs | {"score": 10, "n_cells": 10})
        assert schema.score == schema.n_cells

    @pytest.mark.parametrize("score", [11, 100])
    def test_should_reject_score_exceeding_n_cells(self, impression_kwargs: dict, score: int) -> None:
        """Score greater than n_cells raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(**impression_kwargs | {"score": score, "n_cells": 10})

    @pytest.mark.parametrize("score", [-11, -100])
    def test_should_reject_negative_score(self, impression_kwargs: dict, score: int) -> None:
        """Negative score raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(**impression_kwargs | {"score": score})

    @pytest.mark.parametrize("n_cells", [0, -1, -10])
    def test_should_reject_nonpositive_n_cells(self, impression_kwargs: dict, n_cells: int) -> None:
        """Non-positive n_cells raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(**impression_kwargs | {"n_cells": n_cells})

    def test_should_reject_missing_param(self, impression_kwargs: dict) -> None:
        """Omitting the param field raises ValidationError."""
        kwargs = {k: v for k, v in impression_kwargs.items() if k != "cells"}
        with pytest.raises(ValidationError):
            CalculateLRImpression(**kwargs)  # type: ignore


class TestCalculateLRStriation:
    """Tests for the CalculateLRStriation schema."""

    def test_should_accept_valid_input(self, striation_kwargs: dict) -> None:
        """Valid input with striation LR parameters is accepted."""
        schema = CalculateLRStriation(**striation_kwargs)
        assert schema.score == striation_kwargs["score"]

    @pytest.mark.parametrize("score", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_should_accept_score_in_valid_range(self, striation_kwargs: dict, score: float) -> None:
        """Score accepts any float value in [-1, 1] (full CCF range)."""
        schema = CalculateLRStriation(**striation_kwargs | {"score": score})
        assert schema.score == score

    @pytest.mark.parametrize("score", [-2.0, -1.01, 1.01, 2.0])
    def test_should_reject_score_outside_ccf_range(self, striation_kwargs: dict, score: float) -> None:
        """Scores outside [-1, 1] raise ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRStriation(**striation_kwargs | {"score": score})
