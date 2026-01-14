import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import HttpUrl, ValidationError

from extractors.schemas import ProcessedDataAccess


@pytest.mark.parametrize("application_protocol", ["http", "https"])
def test_processed_data_access_valid_urls(application_protocol: str) -> None:
    """Test that ProcessedDataAccess accepts valid HTTP and HTTPS URLs."""
    # Arrange
    x3p_url = f"{application_protocol}://localhost:8000/extractor/files/token123/scan.x3p"
    preview_url = f"{application_protocol}://localhost:8000/extractor/files/token123/preview.png"
    surface_map_url = f"{application_protocol}://localhost:8000/extractor/files/token123/surface_map.png"
    # Act
    processed_data = ProcessedDataAccess(
        scan=x3p_url,  # type: ignore
        preview=preview_url,  # type: ignore
        surface_map=surface_map_url,  # type: ignore
    )

    # Assert
    assert processed_data.scan_image == HttpUrl(x3p_url)
    assert processed_data.preview_image == HttpUrl(preview_url)
    assert processed_data.surface_map_image == HttpUrl(surface_map_url)


@pytest.mark.parametrize(
    ("field_name", "alias_name"),
    [("x3p_image", "scan"), ("preview_image", "preview"), ("surface_map_image", "surface_map")],
)
@given(
    invalid_url=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "P"), max_codepoint=127),
        min_size=1,
        max_size=50,
    ).filter(lambda x: not x.startswith(("http://", "https://")))
)
def test_processed_data_access_invalid_urls(field_name: str, alias_name: str, invalid_url: str) -> None:
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
            scan=urls["x3p_image"],  # type: ignore
            preview=urls["preview_image"],  # type: ignore
            surface_map=urls["surface_map_image"],  # type: ignore
        )
    assert alias_name in str(exc_info.value)


def test_model_is_frozen() -> None:
    """Test that ProcessedDataAccess instances are immutable (frozen)."""
    # Arrange
    processed_data = ProcessedDataAccess(
        scan="http://localhost:8000/scan.x3p",  # type: ignore
        preview="http://localhost:8000/preview.png",  # type: ignore
        surface_map="http://localhost:8000/surface_map.png",  # type: ignore
    )

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        processed_data.scan_image = "http://localhost:8000/new_scan.x3p"  # type: ignore[misc]
    assert "Instance is frozen" in str(exc_info.value)


def test_extra_fields_forbidden() -> None:
    """Test that extra fields are not allowed."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ProcessedDataAccess(
            scan="http://localhost:8000/scan.x3p",
            preview="http://localhost:8000/preview.png",
            surface_map="http://localhost:8000/surface_map.png",
            extra_field="not allowed",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_missing_required_field() -> None:
    """Test that missing required fields raise ValidationError."""
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ProcessedDataAccess(
            scan="http://localhost:8000/scan.x3p",
            preview="http://localhost:8000/preview.png",
            # Missing surface_map
        )  # type: ignore[call-arg]
    assert "surface_map" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)
