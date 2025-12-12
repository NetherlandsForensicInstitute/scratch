import pytest
from pydantic import HttpUrl

from preprocessors import ProcessedDataLocation


@pytest.fixture(scope="module")
def process_scan_files() -> dict[str, HttpUrl]:
    """Create temporary image files in the same directory."""
    return {
        "x3p_image": HttpUrl("http://something/x3p_image.x3p"),
        "preview_image": HttpUrl("http://something/preview_image.png"),
        "surfacemap_image": HttpUrl("http://something/surfacemap_image.png"),
    }


def test_processed_data_is_url(process_scan_files: dict[str, HttpUrl]) -> None:
    """Test that ProcessedDataLocation accepts files from the same parent directory."""
    # Act
    process_scan = ProcessedDataLocation(**process_scan_files)

    # Assert
    assert isinstance(process_scan.x3p_image, HttpUrl)
    assert isinstance(process_scan.preview_image, HttpUrl)
    assert isinstance(process_scan.surfacemap_image, HttpUrl)
