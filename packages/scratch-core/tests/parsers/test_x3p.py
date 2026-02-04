from re import compile, escape
from pathlib import Path
from unittest.mock import patch

import pytest
from returns.io import IOSuccess, IOFailure
from returns.result import Failure

from container_models import ImageContainer
from parsers import parse_to_x3p
from renders.image_io import save_x3p
from x3p import X3Pfile


def is_good_fail_logs(message: str, log: str) -> bool:
    log_pattern = compile(
        rf"DEBUG.*{escape(message)}:[\s\S]*?"
        rf"ERROR.*{escape(message)}"
    )
    return log_pattern.search(log) is not None


@pytest.mark.parametrize(
    "function",
    (
        "_set_record1_entries",
        "_set_record2_entries",
        "_set_record3_entries",
        "_set_binary_data",
    ),
)
class TestParseToX3PFailure:
    def test_parse_to_x3p_returns_failure(
        self, function: str, image_container: ImageContainer
    ):
        """Test that parse_to_x3p returns Failure when sub-functions fails."""
        with patch(f"parsers.x3p.{function}") as mocker:
            mocker.side_effect = RuntimeError("Some Error")
            result = parse_to_x3p(image_container)

        assert isinstance(result, Failure)
        assert isinstance(result.failure(), RuntimeError)

    def test_parse_to_x3p_logs_on_failure(
        self,
        function: str,
        image_container: ImageContainer,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that parse_to_x3p logs when sub-functions fails."""
        with patch(f"parsers.x3p.{function}") as mocker:
            mocker.side_effect = RuntimeError("Some Error")
            with caplog.at_level("DEBUG"):
                _ = parse_to_x3p(image_container)

        assert is_good_fail_logs("Failed to parse image X3P", caplog.text), (
            "Logs don't match expected format."
        )


class TestX3PSave:
    @pytest.fixture
    def x3p(self, image_container: ImageContainer) -> X3Pfile:
        return parse_to_x3p(image_container).unwrap()

    def test_save_to_x3p_returns_failure_when_write_fails(self, x3p: X3Pfile):
        """Test that save returns IOFailure when write operation fails."""

        # Use a path that will cause write to fail (read-only or invalid)
        result = save_x3p(x3p, output_path=Path("nonexistent_dir/test.x3p"))
        assert isinstance(result, IOFailure)
        assert "No such file or directory" in str(result.failure())

    def test_save_x3p_logs_on_failure(
        self, x3p: X3Pfile, caplog: pytest.LogCaptureFixture
    ):
        """Test that save logs when io fails."""
        output_path = Path("nonexistent_dir/test.x3p")

        with caplog.at_level("DEBUG"):
            _ = save_x3p(x3p, output_path)

        assert is_good_fail_logs("Failed to write X3P file", caplog.text), (
            "Logs don't match expected format."
        )

    def test_save_x3p_logs_on_success(
        self, x3p: X3Pfile, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Test that save logs on happy path."""
        output_path = tmp_path / "test.x3p"

        with caplog.at_level("INFO"):
            _ = save_x3p(x3p, output_path)

        assert compile("Successfully written X3P").search(caplog.text), (
            "Logs don't match expected format."
        )

    def test_save_x3p_returns_success_on_valid_input(
        self, x3p: X3Pfile, tmp_path: Path
    ):
        """Test that save_to_x3p returns IOSuccess(None) when save succeeds."""
        output_path = tmp_path / "test.x3p"
        result = save_x3p(x3p, output_path=output_path)
        assert result == IOSuccess(output_path)
        assert output_path.exists()


def test_parse_to_x3p_on_success(
    caplog: pytest.LogCaptureFixture, image_container: ImageContainer
):
    """Test that parse_to_x3p logs INFO on successful parsing."""
    with caplog.at_level("INFO"):
        result = parse_to_x3p(image_container)

    # TODO: How do I test that X3P is a valid object?
    assert isinstance(result.unwrap(), X3Pfile)


def test_parse_to_x3p_logs_on_success(
    caplog: pytest.LogCaptureFixture, image_container: ImageContainer
):
    """Test that parse_to_x3p logs INFO on successful parsing."""
    with caplog.at_level("INFO"):
        _ = parse_to_x3p(image_container)

    assert compile("Successfully parse array to x3p").search(caplog.text), (
        "Logs don't match expected format."
    )
