from re import compile, escape
from unittest.mock import patch

import pytest
from returns.result import Failure
from x3p import X3Pfile

from container_models.image import ImageContainer
from parsers.x3p import parse_to_x3p


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
        self, function: str, process_image: ImageContainer
    ):
        """Test that parse_to_x3p returns Failure when sub-functions fails."""
        with patch(f"parsers.x3p.{function}") as mocker:
            mocker.side_effect = RuntimeError("Some Error")
            result = parse_to_x3p(process_image)

        assert isinstance(result, Failure)
        assert isinstance(result.failure(), RuntimeError)

    def test_parse_to_x3p_logs_on_failure(
        self,
        function: str,
        process_image: ImageContainer,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that parse_to_x3p logs when sub-functions fails."""
        with patch(f"parsers.x3p.{function}") as mocker:
            mocker.side_effect = RuntimeError("Some Error")
            with caplog.at_level("DEBUG"):
                _ = parse_to_x3p(process_image)

        assert is_good_fail_logs("Failed to parse image X3P", caplog.text), (
            "Logs don't match expected format."
        )


class TestParseToX3PSuccess:
    def test_parse_to_x3p_on_success(
        self, caplog: pytest.LogCaptureFixture, process_image: ImageContainer
    ):
        """Test that parse_to_x3p logs INFO on successful parsing."""
        with caplog.at_level("INFO"):
            result = parse_to_x3p(process_image)

        # TODO: How do I test that X3P is a valid object?
        assert isinstance(result.unwrap(), X3Pfile)

    def test_parse_to_x3p_logs_on_success(
        self, caplog: pytest.LogCaptureFixture, process_image: ImageContainer
    ):
        """Test that parse_to_x3p logs INFO on successful parsing."""
        with caplog.at_level("INFO"):
            _ = parse_to_x3p(process_image)

        assert compile("Successfully parse array to x3p").search(caplog.text), (
            "Logs don't match expected format."
        )
