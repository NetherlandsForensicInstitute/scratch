import pytest
from pydantic import HttpUrl, ValidationError

from preprocessors.schemas import ProcessDataUrls


class TestProcessDataUrls:
    """Tests for ProcessDataUrls immutable URL collection model."""

    @pytest.mark.parametrize(
        "urls",
        [
            pytest.param(
                ("http://localhost:8000/extractor/files/abc123/project/scan.x3p",),
                id="single URL",
            ),
            pytest.param(
                (
                    "http://127.0.0.1:8000/extractor/files/token/project/scan.x3p",
                    "https://localhost:8000/extractor/files/token123/project/surface_map.png",
                    "http://localhost:8000/extractor/files/token123/project/preview.png",
                ),
                id="multiple URLs",
            ),
        ],
    )
    def test_should_create_with_urls(self, urls: tuple[str, ...]) -> None:
        """Test that ProcessDataUrls can be created with single or multiple URLs."""
        # Act
        process_urls = ProcessDataUrls(urls)  # type: ignore

        # Assert
        assert all(isinstance(url, HttpUrl) for url in process_urls.root)
        assert tuple(str(url) for url in process_urls.root) == urls

    def test_should_reject_invalid_url_format(self) -> None:
        """Test that ProcessDataUrls rejects invalid URL formats."""
        # Act & Assert
        with pytest.raises(ValidationError, match="url") as exc_info:
            ProcessDataUrls(("not-a-valid-url",))  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any("url" in error["type"].lower() for error in errors)

    @pytest.mark.parametrize(
        "invalid_url",
        [
            pytest.param("ftp://example.com/file.x3p", id="ftp scheme"),
            pytest.param("file:///path/to/file.x3p", id="file scheme"),
            pytest.param("mailto:user@example.com", id="mailto scheme"),
        ],
    )
    def test_should_reject_non_http_schemes(self, invalid_url: str) -> None:
        """Test that ProcessDataUrls rejects non-HTTP(S) URL schemes."""
        # Act & Assert
        with pytest.raises(ValidationError, match="url") as exc_info:
            ProcessDataUrls((invalid_url,))  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any("url" in error["type"].lower() for error in errors)

    def test_should_reject_empty_sequence(self) -> None:
        """Test that ProcessDataUrls rejects empty URL sequences."""
        # Act & Assert
        with pytest.raises(ValidationError, match="at least 1 item") as exc_info:
            ProcessDataUrls(())  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert len(errors) > 0
