import inspect
from enum import StrEnum

import pytest
from response_constants import LRFiles
from pydantic import Field, HttpUrl, ValidationError

import extractors.schemas as schemas_module
from extractors.schemas import (
    SupportedExtension,
)
from response_models import LRResponse, LRResponseURL, URLContainer

_EMPTY_LR_STATS = {
    "lr_lower_ci": None,
    "lr_upper_ci": None,
}

_CONCRETE_CLASSES = [
    cls
    for _, cls in inspect.getmembers(schemas_module, inspect.isclass)
    if cls.__module__ == schemas_module.__name__
    and cls is not URLContainer
    and cls is not SupportedExtension
    and cls is not LRResponse
    and not issubclass(cls, LRResponse)
]

_BASE_URL = "http://localhost:8000/files/token123"


@pytest.mark.parametrize("cls", _CONCRETE_CLASSES, ids=lambda cls: cls.__name__)
def test_all_module_classes_extend_base_response_urls(cls: type[URLContainer]) -> None:
    """Every model class in the schemas module inherits from URLContainer."""
    assert issubclass(cls, URLContainer)


class _SimpleURLs(URLContainer):
    """Minimal concrete subclass."""

    some_file: HttpUrl = Field(...)
    plain_file: HttpUrl = Field(...)


class _SimpleFiles(StrEnum):
    some_file = "file.png"
    plain_file = "file.txt"


class _NoFileName(URLContainer):
    """Subclass with a field whose json_schema_extra is missing 'file_name'."""

    broken_field: HttpUrl = Field(..., json_schema_extra={})


class TestGenerateUrls:
    """Tests for URLContainer.generate_urls."""

    @pytest.mark.parametrize("protocol", ["http", "https"])
    def test_supports_http_and_https(self, protocol: str) -> None:
        """Both HTTP and HTTPS base URLs are accepted."""
        base_url = f"{protocol}://localhost:8000/files/token"
        result = _SimpleURLs.from_enum(enum=_SimpleFiles, base_url=base_url)
        assert str(result.some_file).startswith(protocol)

    @pytest.mark.parametrize(
        "invalid_url",
        [
            pytest.param("", id="empty_string"),
            pytest.param("not-a-url", id="non_url_string"),
            pytest.param("ftp://localhost/files", id="non_http_scheme"),
        ],
    )
    def test_raises_validation_error_for_invalid_base_url(self, invalid_url: str) -> None:
        """generate_urls raises ValidationError when the base URL is not a valid HTTP/HTTPS URL."""
        with pytest.raises(ValidationError):
            _SimpleURLs.from_enum(enum=_SimpleFiles, base_url=invalid_url)

    def test_raises_when_enum_has_incorrect_field_namings(self):
        """Should raise error when enum has not the correct values."""

        class WrongFiles(StrEnum):
            wrong_file = "oeps.txt"

        with pytest.raises(ValidationError):
            _SimpleURLs.from_enum(enum=WrongFiles, base_url=_BASE_URL)


class TestLRResponse:
    """Tests for LRResponse serialization."""

    @pytest.fixture
    def lr_response(self) -> LRResponse:
        """LRResponse instance with a valid LRResponseURL and a non-zero lr value."""
        return LRResponse(urls=LRResponseURL.from_enum(enum=LRFiles, base_url=_BASE_URL), llr=2.5, **_EMPTY_LR_STATS)

    def test_serialized_output_is_flat(self, lr_response: LRResponse) -> None:
        """model_dump produces a flat dict with no nested 'urls' key."""
        data = lr_response.model_dump()

        assert "urls" not in data

    def test_url_fields_promoted_to_top_level(self, lr_response: LRResponse) -> None:
        """All fields from LRResponseURL appear at the top level."""
        url_fields = set(LRResponseURL.model_fields)
        data = lr_response.model_dump()

        assert url_fields <= set(data)

    def test_raises_validation_error_for_non_numeric_lr(self) -> None:
        """ValidationError is raised when lr cannot be coerced to float."""
        with pytest.raises(ValidationError):
            LRResponse(
                urls=LRResponseURL.from_enum(enum=LRFiles, base_url=_BASE_URL),
                llr="not-a-number",  # pyright: ignore[reportArgumentType]
                **_EMPTY_LR_STATS,
            )
