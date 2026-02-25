from itertools import chain, combinations
from pathlib import Path

import pytest
from pydantic import ValidationError

from processors.schemas import (
    CalculateLR,
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScoreImpression,
    CalculateScoreStriation,
    ImpressionLRParamaters,
    ImpressionParameters,
    MarkDirectories,
    StriationLRParamaters,
    StriationParamaters,
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


_SCORE_WITH_PARAM_FIELDS = ("mark_ref", "mark_comp", "param")
_SCORE_WITH_PARAM_MISSING_COMBOS = list(
    chain.from_iterable(combinations(_SCORE_WITH_PARAM_FIELDS, r) for r in range(1, len(_SCORE_WITH_PARAM_FIELDS) + 1))
)


class TestCalculateScore:
    """Tests for the CalculateScore schema."""

    def test_should_accept_valid_directories(self, mark_ref: Path, mark_comp: Path) -> None:
        """Test that valid existing directories are accepted."""
        # Act
        schema = MarkDirectories(mark_ref=mark_ref, mark_comp=mark_comp)

        # Assert
        assert schema.mark_ref == mark_ref
        assert schema.mark_comp == mark_comp

    @pytest.mark.parametrize("field", ["mark_ref", "mark_comp"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_ref: Path, mark_comp: Path
    ) -> None:
        """Test that non-existent paths and file paths raise ValidationError."""
        # Arrange
        fields = {"mark_ref": mark_ref, "mark_comp": mark_comp} | {field: invalid_path}

        # Act & Assert
        with pytest.raises(ValidationError):
            MarkDirectories.model_validate(fields)


class TestCalculateLR:
    """Tests for the CalculateLR schema."""

    def test_should_accept_valid_input(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that valid directories, score, and lr_system file are accepted."""
        # Arrange
        score = 42

        # Act
        schema = CalculateLR(mark_ref=mark_ref, mark_comp=mark_comp, score=score, lr_system=lr_system_file)

        # Assert
        assert schema.score == score
        assert schema.lr_system == lr_system_file

    @pytest.mark.parametrize(
        "invalid_lr_system",
        [
            pytest.param(Path() / "nonexistent.bin", id="nonexistent"),
            pytest.param(Path(__file__).parent, id="directory"),
        ],
    )
    def test_should_reject_invalid_lr_system(self, mark_ref: Path, mark_comp: Path, invalid_lr_system: Path) -> None:
        """Test that a non-existent path and a directory path for lr_system raise ValidationError."""
        # Act & Assert
        with pytest.raises(ValidationError):
            CalculateLR(mark_ref=mark_ref, mark_comp=mark_comp, score=0, lr_system=invalid_lr_system)

    @pytest.mark.parametrize("field", ["mark_ref", "mark_comp"])
    @pytest.mark.parametrize("invalid_path", [Path() / "nonexistent", Path(__file__)])
    def test_should_reject_invalid_directory_path(
        self, field: str, invalid_path: Path, mark_ref: Path, mark_comp: Path, lr_system_file: Path
    ) -> None:
        """Test that non-existent paths and file paths raise ValidationError."""
        # Arrange
        fields = {"mark_ref": mark_ref, "mark_comp": mark_comp, "lr_system": lr_system_file, "score": 0} | {
            field: invalid_path
        }

        # Act & Assert
        with pytest.raises(ValidationError):
            CalculateLR.model_validate(fields)

    @pytest.mark.parametrize("score", [-100, -1, 0, 1, 100])
    def test_should_accept_any_integer_score(
        self, mark_ref: Path, mark_comp: Path, lr_system_file: Path, score: int
    ) -> None:
        """Test that score accepts any integer value."""
        # Act
        schema = CalculateLR(mark_ref=mark_ref, mark_comp=mark_comp, score=score, lr_system=lr_system_file)

        # Assert
        assert schema.score == score


class TestCalculateLRImpression:
    """Tests for the CalculateLRImpression schema."""

    def test_should_accept_valid_input(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that valid input including n_cells and impression LR parameters is accepted."""
        # Arrange
        n_cells = 10

        # Act
        schema = CalculateLRImpression(
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            score=5,
            lr_system=lr_system_file,
            n_cells=n_cells,
            param=ImpressionLRParamaters(),
        )

        # Assert
        assert schema.n_cells == n_cells
        assert isinstance(schema.param, ImpressionLRParamaters)

    @pytest.mark.parametrize("n_cells", [0, -1, -10])
    def test_should_reject_nonpositive_n_cells(
        self, mark_ref: Path, mark_comp: Path, lr_system_file: Path, n_cells: int
    ) -> None:
        """Test that non-positive n_cells raises ValidationError."""
        # Act & Assert
        with pytest.raises(ValidationError):
            CalculateLRImpression(
                mark_ref=mark_ref,
                mark_comp=mark_comp,
                score=5,
                lr_system=lr_system_file,
                n_cells=n_cells,
                param=ImpressionLRParamaters(),
            )

    def test_should_reject_missing_param(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that omitting param raises ValidationError."""
        # Act & Assert
        with pytest.raises(ValidationError):
            CalculateLRImpression(  # type: ignore
                mark_ref=mark_ref,
                mark_comp=mark_comp,
                score=5,
                lr_system=lr_system_file,
                n_cells=10,
            )


class TestCalculateLRStriation:
    """Tests for the CalculateLRStriation schema."""

    def test_should_accept_valid_input(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that valid input with striation LR parameters is accepted."""
        # Act
        schema = CalculateLRStriation(
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            score=5,
            lr_system=lr_system_file,
            param=StriationLRParamaters(),
        )

        # Assert
        assert isinstance(schema.param, StriationLRParamaters)

    def test_should_reject_missing_param(self, mark_ref: Path, mark_comp: Path, lr_system_file: Path) -> None:
        """Test that omitting param raises ValidationError."""
        # Act & Assert
        with pytest.raises(ValidationError):
            CalculateLRStriation(  # type: ignore
                mark_ref=mark_ref,
                mark_comp=mark_comp,
                score=5,
                lr_system=lr_system_file,
            )
