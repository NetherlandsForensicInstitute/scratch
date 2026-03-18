from pathlib import Path
from unittest.mock import patch

import pytest
from x3p import X3Pfile

from container_models.scan_image import ScanImage
from parsers import parse_to_x3p, save_x3p


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
    def test_parse_to_x3p_returns_failure(self, function: str, scan_image: ScanImage):
        """Test that parse_to_x3p returns Failure when sub-functions fails."""
        with patch(f"parsers.x3p.{function}") as mocker:
            mocker.side_effect = RuntimeError("Some Error")
            with pytest.raises(RuntimeError):
                parse_to_x3p(scan_image)


class TestX3PSave:
    @pytest.fixture(scope="class")
    def x3p(self, scan_image: ScanImage) -> X3Pfile:
        return parse_to_x3p(scan_image)

    def test_save_to_x3p_returns_failure_when_write_fails(self, x3p: X3Pfile):
        """Test that raises error when write operation fails."""

        # Use a path that will cause write to fail (read-only or invalid)
        with pytest.raises(FileNotFoundError):
            save_x3p(x3p, output_path=Path("nonexistent_dir/test.x3p"))

    def test_save_x3p_returns_success_on_valid_input(
        self, x3p: X3Pfile, tmp_path: Path
    ):
        """Test that save_to_x3p returns IOSuccess(None) when save succeeds."""
        output_path = tmp_path / "test.x3p"
        save_x3p(x3p, output_path=output_path)
        assert output_path.exists()


def test_parse_to_x3p_on_success(scan_image: ScanImage):
    """Test that parse_to_x3p logs INFO on successful parsing."""
    result = parse_to_x3p(scan_image)

    # TODO: How do I test that X3P is a valid object?
    assert isinstance(result, X3Pfile)


def test_parse_to_x3p_logs_on_success(scan_image: ScanImage):
    """Test that parse_to_x3p logs INFO on successful parsing."""
    _ = parse_to_x3p(scan_image)
