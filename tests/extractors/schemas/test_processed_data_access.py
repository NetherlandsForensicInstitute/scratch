import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import HttpUrl, ValidationError

from extractors.schemas import ProcessedDataAccess


class MockProcessData:
    """Mock ProcessData for testing."""

    @property
    def x3p_filename(self) -> str:
        return "test_scan.x3p"

    @property
    def preview_filename(self) -> str:
        return "test_scan_preview.png"

    @property
    def surface_map_filename(self) -> str:
        return "test_scan_surface_map.png"


@pytest.mark.parametrize("application_protocol", ["http", "https"])
def test_processed_data_access_valid_urls(application_protocol: str) -> None:
    """Test that ProcessedDataAccess accepts valid HTTP and HTTPS URLs."""
    # Arrange
    x3p_url = f"{application_protocol}://localhost:8000/extractor/files/token123/scan.x3p"
    preview_url = f"{application_protocol}://localhost:8000/extractor/files/token123/preview.png"
    surface_map_url = f"{application_protocol}://localhost:8000/extractor/files/token123/surface_map.png"
    # Act
    processed_data = ProcessedDataAccess(
        x3p_image=x3p_url,  # type: ignore
        preview_image=preview_url,  # type: ignore
        surface_map_image=surface_map_url,  # type: ignore
    )

    # Assert
    assert processed_data.x3p_image == HttpUrl(x3p_url)
    assert processed_data.preview_image == HttpUrl(preview_url)
    assert processed_data.surface_map_image == HttpUrl(surface_map_url)


@pytest.mark.parametrize("field_name", ["x3p_image", "preview_image", "surface_map_image"])
@given(
    invalid_url=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "P"), max_codepoint=127),
        min_size=1,
        max_size=50,
    ).filter(lambda x: not x.startswith(("http://", "https://")))
)
def test_processed_data_access_invalid_urls(field_name: str, invalid_url: str) -> None:
    """Test that invalid URLs raise ValidationError using property-based testing."""
    # Arrange
    valid_urls = {
        "x3p_image": "http://localhost:8000/scan.x3p",
        "preview_image": "http://localhost:8000/preview.png",
        "surface_map_image": "http://localhost:8000/surface_map.png",
    }

    # Replace the field being tested with the invalid URL
    urls = valid_urls.copy()
    urls[field_name] = invalid_url  # type: ignore

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ProcessedDataAccess(
            x3p_image=urls["x3p_image"],  # type: ignore
            preview_image=urls["preview_image"],  # type: ignore
            surface_map_image=urls["surface_map_image"],  # type: ignore
        )
    assert field_name in str(exc_info.value)


def test_from_access_point() -> None:
    """Test that from_access_point creates ProcessedDataAccess correctly."""
    # Arrange
    access_url = "http://localhost:8000/extractor/files/token123"
    process_data = MockProcessData()

    # Act
    result = ProcessedDataAccess.from_access_point(access_url, process_data)

    # Assert
    assert isinstance(result, ProcessedDataAccess)
    assert str(result.x3p_image) == f"{access_url}/test_scan.x3p"
    assert str(result.preview_image) == f"{access_url}/test_scan_preview.png"
    assert str(result.surface_map_image) == f"{access_url}/test_scan_surface_map.png"


def test_from_access_point_with_trailing_slash() -> None:
    """Test that from_access_point handles URLs without requiring trailing slash."""
    # Arrange
    access_url = "http://localhost:8000/extractor/files/token123"
    process_data = MockProcessData()

    # Act
    result = ProcessedDataAccess.from_access_point(access_url, process_data)

    # Assert
    # Should not have double slashes
    assert "//" not in str(result.x3p_image).replace("://", "")
    assert "//" not in str(result.preview_image).replace("://", "")
    assert "//" not in str(result.surface_map_image).replace("://", "")


def test_model_is_frozen() -> None:
    """Test that ProcessedDataAccess instances are immutable (frozen)."""
    # Arrange
    processed_data = ProcessedDataAccess(
        x3p_image="http://localhost:8000/scan.x3p",  # type: ignore
        preview_image="http://localhost:8000/preview.png",  # type: ignore
        surface_map_image="http://localhost:8000/surface_map.png",  # type: ignore
    )

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        processed_data.x3p_image = "http://localhost:8000/new_scan.x3p"  # type: ignore[misc]
    assert "Instance is frozen" in str(exc_info.value)


def test_extra_fields_forbidden() -> None:
    """Test that extra fields are not allowed."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ProcessedDataAccess(
            x3p_image="http://localhost:8000/scan.x3p",
            preview_image="http://localhost:8000/preview.png",
            surface_map_image="http://localhost:8000/surface_map.png",
            extra_field="not allowed",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_missing_required_field() -> None:
    """Test that missing required fields raise ValidationError."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ProcessedDataAccess(
            x3p_image="http://localhost:8000/scan.x3p",
            preview_image="http://localhost:8000/preview.png",
            # Missing surface_map_image
        )  # type: ignore[call-arg]
    assert "surface_map_image" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)
