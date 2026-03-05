from pathlib import Path

import pytest
from pydantic import ValidationError

from processors.schemas import (
    CalculateLR,
    CalculateLRImpression,
    CalculateLRStriation,
    ImpressionLRParamaters,
    MarkDirectories,
    StriationLRParamaters,
)


@pytest.fixture
def mark_ref(tmp_path: Path) -> Path:
    directory = tmp_path / "mark_ref"
    directory.mkdir()
    return directory


@pytest.fixture
def mark_comp(tmp_path: Path) -> Path:
    directory = tmp_path / "mark_comp"
    directory.mkdir()
    return directory


@pytest.fixture
def lr_system_file(tmp_path: Path) -> Path:
    f = tmp_path / "lr_system.bin"
    f.touch()
    return f


class TestMarkDirectories:
    """Tests for the MarkDirectories schema."""

    def test_should_accept_valid_directories(self, mark_ref: Path, mark_comp: Path) -> None:
        """Test that valid existing directories are accepted."""
        schema = MarkDirectories(mark_dir_ref=mark_ref, mark_dir_comp=mark_comp)

        assert schema.mark_dir_ref == mark_ref
        assert schema.mark_dir_comp == mark_comp

    @pytest.mark.parametrize("field", ["mark_dir_ref", "mark_dir_comp"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_ref: Path, mark_comp: Path
    ) -> None:
        """Test that non-existent paths and file paths raise ValidationError."""
        fields = {"mark_dir_ref": mark_ref, "mark_dir_comp": mark_comp} | {field: invalid_path}

        with pytest.raises(ValidationError):
            MarkDirectories.model_validate(fields)


class TestCalculateLR:
    """Tests for the CalculateLR schema."""

    def test_should_accept_valid_input(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that valid directories and lr_system_path file are accepted."""
        schema = CalculateLR(mark_dir_ref=mark_ref, mark_dir_comp=mark_comp, lr_system_path=lr_system_file)

        assert schema.lr_system_path == lr_system_file

    @pytest.mark.parametrize(
        "invalid_lr_system",
        [
            pytest.param(Path() / "nonexistent.bin", id="nonexistent"),
            pytest.param(Path(__file__).parent, id="directory"),
        ],
    )
    def test_should_reject_invalid_lr_system_path(
        self, mark_ref: Path, mark_comp: Path, invalid_lr_system: Path
    ) -> None:
        """Test that a non-existent path and a directory path for lr_system_path raise ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLR(mark_dir_ref=mark_ref, mark_dir_comp=mark_comp, lr_system_path=invalid_lr_system)

    @pytest.mark.parametrize("field", ["mark_dir_ref", "mark_dir_comp"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_ref: Path, mark_comp: Path, lr_system_file: Path
    ) -> None:
        """Test that non-existent paths and file paths raise ValidationError."""
        fields = {"mark_dir_ref": mark_ref, "mark_dir_comp": mark_comp, "lr_system_path": lr_system_file} | {
            field: invalid_path
        }

        with pytest.raises(ValidationError):
            CalculateLR.model_validate(fields)


class TestCalculateLRImpression:
    """Tests for the CalculateLRImpression schema."""

    def test_should_accept_valid_input(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that valid input including n_cells and impression LR parameters is accepted."""
        score = 5
        n_cells = 10
        schema = CalculateLRImpression(
            mark_dir_ref=mark_ref,
            mark_dir_comp=mark_comp,
            score=score,
            lr_system_path=lr_system_file,
            n_cells=n_cells,
            param=ImpressionLRParamaters(),
        )

        assert schema.score == score
        assert schema.n_cells == n_cells
        assert isinstance(schema.param, ImpressionLRParamaters)

    @pytest.mark.parametrize("score", [0, 1, 100])
    def test_should_accept_score_within_n_cells(
        self, mark_ref: Path, mark_comp: Path, lr_system_file: Path, score: int
    ) -> None:
        """Test that score accepts any integer value up to n_cells."""
        schema = CalculateLRImpression(
            mark_dir_ref=mark_ref,
            mark_dir_comp=mark_comp,
            score=score,
            lr_system_path=lr_system_file,
            n_cells=100,
            param=ImpressionLRParamaters(),
        )

        assert schema.score == score

    def test_should_accept_score_equal_to_n_cells(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that score equal to n_cells is valid (all cells match)."""
        schema = CalculateLRImpression(
            mark_dir_ref=mark_ref,
            mark_dir_comp=mark_comp,
            score=10,
            lr_system_path=lr_system_file,
            n_cells=10,
            param=ImpressionLRParamaters(),
        )

        assert schema.score == schema.n_cells

    @pytest.mark.parametrize("score", [11, 100])
    def test_should_reject_score_exceeding_n_cells(
        self, mark_ref: Path, mark_comp: Path, lr_system_file: Path, score: int
    ) -> None:
        """Test that score greater than n_cells raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(
                mark_dir_ref=mark_ref,
                mark_dir_comp=mark_comp,
                score=score,
                lr_system_path=lr_system_file,
                n_cells=10,
                param=ImpressionLRParamaters(),
            )

    @pytest.mark.parametrize("n_cells", [0, -1, -10])
    def test_should_reject_nonpositive_n_cells(
        self, mark_ref: Path, mark_comp: Path, lr_system_file: Path, n_cells: int
    ) -> None:
        """Test that non-positive n_cells raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(
                mark_dir_ref=mark_ref,
                mark_dir_comp=mark_comp,
                score=5,
                lr_system_path=lr_system_file,
                n_cells=n_cells,
                param=ImpressionLRParamaters(),
            )

    def test_should_reject_missing_param(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that omitting param raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRImpression(  # type: ignore
                mark_dir_ref=mark_ref,
                mark_dir_comp=mark_comp,
                score=5,
                lr_system_path=lr_system_file,
                n_cells=10,
            )


class TestCalculateLRStriation:
    """Tests for the CalculateLRStriation schema."""

    def test_should_accept_valid_input(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that valid input with striation LR parameters is accepted."""
        score = 0.5
        schema = CalculateLRStriation(
            mark_dir_ref=mark_ref,
            mark_dir_comp=mark_comp,
            score=score,
            lr_system_path=lr_system_file,
            param=StriationLRParamaters(),
        )

        assert schema.score == score
        assert isinstance(schema.param, StriationLRParamaters)

    @pytest.mark.parametrize("score", [-1.0, 0.0, 0.5, 1.0])
    def test_should_accept_any_float_score(
        self, mark_ref: Path, mark_comp: Path, lr_system_file: Path, score: float
    ) -> None:
        """Test that score accepts any float value."""
        schema = CalculateLRStriation(
            mark_dir_ref=mark_ref,
            mark_dir_comp=mark_comp,
            score=score,
            lr_system_path=lr_system_file,
            param=StriationLRParamaters(),
        )

        assert schema.score == score

    def test_should_reject_missing_param(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that omitting param raises ValidationError."""
        with pytest.raises(ValidationError):
            CalculateLRStriation(  # type: ignore
                mark_dir_ref=mark_ref,
                mark_dir_comp=mark_comp,
                score=0.5,
                lr_system_path=lr_system_file,
            )
