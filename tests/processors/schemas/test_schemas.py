from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from processors.schemas import (
    CalculateLR,
    CalculateLRImpression,
    CalculateLRStriation,
    ImpressionLRParameters,
    MarkDirectories,
    StriationLRParameters,
)


class TestMarkDirectories:
    """Tests for the MarkDirectories schema."""

    def test_should_accept_valid_directories(self, mark_dir_ref: Path, mark_dir_comp: Path) -> None:
        """Test that valid existing directories are accepted."""
        schema = MarkDirectories(mark_dir_ref=mark_dir_ref, mark_dir_comp=mark_dir_comp)

        assert schema.mark_dir_ref == mark_dir_ref
        assert schema.mark_dir_comp == mark_dir_comp

    @pytest.mark.parametrize("field", ["mark_dir_ref", "mark_dir_comp"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_dir_ref: Path, mark_dir_comp: Path
    ) -> None:
        """Test that non-existent paths and file paths raise ValidationError."""
        fields = {"mark_dir_ref": mark_dir_ref, "mark_dir_comp": mark_dir_comp} | {field: invalid_path}

        with pytest.raises(ValidationError):
            MarkDirectories.model_validate(fields)


class TestCalculateLR:
    """Tests for the CalculateLR schema."""

    def test_should_accept_valid_input(self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path) -> None:
        """Test that valid directories and lr_system_path file are accepted."""
        schema = CalculateLR(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            lr_system_path=lr_system_file,
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
        )

        assert schema.lr_system_path == lr_system_file

    @pytest.mark.parametrize(
        "invalid_lr_system",
        [
            pytest.param(Path() / "nonexistent.bin", id="nonexistent"),
            pytest.param(Path(__file__).parent, id="directory"),
        ],
    )
    def test_should_reject_invalid_lr_system_path(
        self, mark_dir_ref: Path, mark_dir_comp: Path, invalid_lr_system: Path
    ) -> None:
        """Test that a non-existent path and a directory path for lr_system_path raise ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLR(
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                lr_system_path=invalid_lr_system,
                user_id="AAAAA",
                date_report=date(2000, 1, 1),
            )

    @pytest.mark.parametrize("field", ["mark_dir_ref", "mark_dir_comp"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path
    ) -> None:
        """Test that non-existent paths and file paths raise ValidationError."""
        fields = {"mark_dir_ref": mark_dir_ref, "mark_dir_comp": mark_dir_comp, "lr_system_path": lr_system_file} | {
            field: invalid_path
        }

        with pytest.raises(ValidationError):
            CalculateLR.model_validate(fields)


class TestCalculateLRImpression:
    """Tests for the CalculateLRImpression schema."""

    def test_should_accept_valid_input(self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path) -> None:
        """Test that valid input including n_cells and impression LR parameters is accepted."""
        score = 5
        n_cells = 10
        schema = CalculateLRImpression(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            score=score,
            lr_system_path=lr_system_file,
            n_cells=n_cells,
            param=ImpressionLRParameters(),
            metadata_compared={"metadata": "compared"},
            metadata_reference={"metadata": "reference"},
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
        )

        assert schema.score == score
        assert schema.n_cells == n_cells
        assert isinstance(schema.param, ImpressionLRParameters)

    @pytest.mark.parametrize("score", [0, 1, 100])
    def test_should_accept_score_within_n_cells(
        self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path, score: int
    ) -> None:
        """Test that score accepts any integer value up to n_cells."""
        schema = CalculateLRImpression(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            score=score,
            lr_system_path=lr_system_file,
            n_cells=100,
            param=ImpressionLRParameters(),
            metadata_compared={"metadata": "compared"},
            metadata_reference={"metadata": "reference"},
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
        )

        assert schema.score == score

    def test_should_accept_score_equal_to_n_cells(
        self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path
    ) -> None:
        """Test that score equal to n_cells is valid (all cells match)."""
        schema = CalculateLRImpression(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            score=10,
            lr_system_path=lr_system_file,
            n_cells=10,
            param=ImpressionLRParameters(),
            metadata_compared={"metadata": "compared"},
            metadata_reference={"metadata": "reference"},
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
        )

        assert schema.score == schema.n_cells

    @pytest.mark.parametrize("score", [11, 100])
    def test_should_reject_score_exceeding_n_cells(
        self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path, score: int
    ) -> None:
        """Test that score greater than n_cells raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                score=score,
                lr_system_path=lr_system_file,
                n_cells=10,
                param=ImpressionLRParameters(),
                metadata_compared={"metadata": "compared"},
                metadata_reference={"metadata": "reference"},
                user_id="AAAAA",
                date_report=date(2000, 1, 1),
            )

    @pytest.mark.parametrize("score", [-11, -100])
    def test_should_reject_negative_score(
        self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path, score: int
    ) -> None:
        """Test that negative score raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                score=score,
                lr_system_path=lr_system_file,
                n_cells=10,
                param=ImpressionLRParameters(),
                metadata_compared={"metadata": "compared"},
                metadata_reference={"metadata": "reference"},
                user_id="AAAAA",
                date_report=date(2000, 1, 1),
            )

    @pytest.mark.parametrize("n_cells", [0, -1, -10])
    def test_should_reject_nonpositive_n_cells(
        self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path, n_cells: int
    ) -> None:
        """Test that non-positive n_cells raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                score=5,
                lr_system_path=lr_system_file,
                n_cells=n_cells,
                param=ImpressionLRParameters(),
                metadata_compared={"metadata": "compared"},
                metadata_reference={"metadata": "reference"},
                user_id="AAAAA",
                date_report=date(2000, 1, 1),
            )

    def test_should_reject_missing_param(self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path) -> None:
        """Test that omitting param raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(  # type: ignore
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                score=5,
                lr_system_path=lr_system_file,
                n_cells=10,
                metadata_compared={"metadata": "compared"},
                metadata_reference={"metadata": "reference"},
                user_id="AAAAA",
                date_report=date(2000, 1, 1),
            )


class TestCalculateLRStriation:
    """Tests for the CalculateLRStriation schema."""

    def test_should_accept_valid_input(self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path) -> None:
        """Test that valid input with striation LR parameters is accepted."""
        score = 0.5
        schema = CalculateLRStriation(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            score=score,
            lr_system_path=lr_system_file,
            param=StriationLRParameters(),
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
            metadata_compared={"metadata": "compared"},
            metadata_reference={"metadata": "reference"},
        )

        assert schema.score == score
        assert isinstance(schema.param, StriationLRParameters)

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_should_accept_positive_float_score(
        self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path, score: float
    ) -> None:
        """Test that score accepts any float value."""
        schema = CalculateLRStriation(
            mark_dir_ref=mark_dir_ref,
            mark_dir_comp=mark_dir_comp,
            score=score,
            lr_system_path=lr_system_file,
            param=StriationLRParameters(),
            user_id="AAAAA",
            date_report=date(2000, 1, 1),
            metadata_compared={"metadata": "compared"},
            metadata_reference={"metadata": "reference"},
        )

        assert schema.score == score

    @pytest.mark.parametrize("score", [-1.0, -0.5, -2.0])
    def test_should_reject_negative_float_score(
        self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path, score: float
    ) -> None:
        """Test that score accepts any float value."""
        with pytest.raises(ValidationError):
            CalculateLRStriation(
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                score=score,
                lr_system_path=lr_system_file,
                param=StriationLRParameters(),
                user_id="AAAAA",
                date_report=date(2000, 1, 1),
            )

    def test_should_reject_missing_param(self, mark_dir_ref: Path, mark_dir_comp: Path, lr_system_file: Path) -> None:
        """Test that omitting param raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRStriation(  # type: ignore
                mark_dir_ref=mark_dir_ref,
                mark_dir_comp=mark_dir_comp,
                score=0.5,
                lr_system_path=lr_system_file,
                user_id="AAAAA",
                date_report=date(2000, 1, 1),
            )
