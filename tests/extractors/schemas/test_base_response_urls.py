import inspect
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import Field, HttpUrl, ValidationError

import extractors.schemas as schemas_module
from extractors.schemas import (
    ComparisonResponseImpression,
    ComparisonResponseStriation,
    LRResponse,
    LRResponseURL,
    PrepareMarkResponseImpression,
    PrepareMarkResponseStriation,
    ProcessedDataAccess,
    SupportedExtension,
    URLContainer,
)

_CONCRETE_CLASSES = [
    cls
    for _, cls in inspect.getmembers(schemas_module, inspect.isclass)
    if cls.__module__ == schemas_module.__name__
    and cls is not URLContainer
    and cls is not SupportedExtension
    and cls is not LRResponse
]

_BASE_URL = "http://localhost:8000/files/token123"


@pytest.mark.parametrize("cls", _CONCRETE_CLASSES, ids=lambda cls: cls.__name__)
def test_all_module_classes_extend_base_response_urls(cls: type[URLContainer]) -> None:
    """Every model class in the schemas module inherits from URLContainer."""
    assert issubclass(cls, URLContainer)


class _SimpleURLs(URLContainer):
    """Minimal concrete subclass with one aliased and one plain field."""

    aliased_file: HttpUrl = Field(...)
    plain_file: HttpUrl = Field(...)


class _SimpleFiles(StrEnum):
    aliased_file = "file.png"
    plain_file = "file.txt"


class _NoFileName(URLContainer):
    """Subclass with a field whose json_schema_extra is missing 'file_name'."""

    broken_field: HttpUrl = Field(..., json_schema_extra={})


class TestGenerateUrls:
    """Tests for URLContainer.generate_urls."""

    def test_aliased_field_populated_via_alias(self) -> None:
        """Aliased fields are set using their alias key."""
        result = _SimpleURLs.from_enum(enum=_SimpleFiles, base_url=_BASE_URL)
        assert result.aliased_file == HttpUrl(f"{_BASE_URL}/{_SimpleFiles.aliased_file}")
        assert result.plain_file == HttpUrl(f"{_BASE_URL}/{_SimpleFiles.plain_file}")

    @pytest.mark.parametrize("protocol", ["http", "https"])
    def test_supports_http_and_https(self, protocol: str) -> None:
        """Both HTTP and HTTPS base URLs are accepted."""
        base_url = f"{protocol}://localhost:8000/files/token"
        result = _SimpleURLs.from_enum(enum=_SimpleFiles, base_url=base_url)
        assert str(result.aliased_file).startswith(protocol)

    def test_all_fields_are_represented(self) -> None:
        """The returned dict contains one entry per model field."""
        fields = set(_SimpleURLs.model_json_schema(by_alias=True)["properties"])
        urls = _SimpleURLs.from_enum(enum=_SimpleFiles, base_url=_BASE_URL).model_dump(by_alias=True)

        assert not (missing := (set(urls) - fields)), f"files {', '.join(missing)} are missing"

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
        """Should raise error when enum has not the correct values"""

        class WrongFiles(StrEnum):
            wrong_file = "oeps.txt"

        with pytest.raises(ValidationError):
            _SimpleURLs.from_enum(enum=WrongFiles, base_url=_BASE_URL)


class TestLRResponse:
    """Tests for LRResponse serialization."""

    @pytest.fixture
    def lr_response(self) -> LRResponse:
        """LRResponse instance with a valid LRResponseURL and a non-zero lr value."""
        return LRResponse(urls=LRResponseURL.generate_urls(_BASE_URL), lr=2.5)

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
            LRResponse(urls=LRResponseURL.generate_urls(_BASE_URL), lr="not-a-number")  # type: ignore


@pytest.mark.parametrize(
    ("klass", "fields"),
    [
        pytest.param(ProcessedDataAccess, {"preview", "surface_map", "scan"}, id=ProcessedDataAccess.__name__),
        pytest.param(
            PrepareMarkResponseStriation,
            {
                "preview",
                "surface_map",
                "mark_data",
                "mark_meta",
                "processed_data",
                "processed_meta",
                "profile_data",
            },
            id=PrepareMarkResponseStriation.__name__,
        ),
        pytest.param(
            PrepareMarkResponseImpression,
            {
                "preview",
                "surface_map",
                "mark_data",
                "mark_meta",
                "processed_data",
                "processed_meta",
                "leveled_data",
                "leveled_meta",
            },
            id=PrepareMarkResponseImpression.__name__,
        ),
        pytest.param(
            ComparisonResponseStriation,
            {
                "mark_ref_surfacemap",
                "mark_comp_surfacemap",
                "filtered_reference_heatmap",
                "comparison_overview",
                "mark_ref_depthmap",
                "mark_comp_depthmap",
                "similarity_plot",
                "filtered_compared_heatmap",
                "side_by_side_heatmap",
                "wavedlength_plot",
            },
            id=ComparisonResponseStriation.__name__,
        ),
        pytest.param(
            ComparisonResponseImpression,
            {
                "mark_ref_surfacemap",
                "mark_comp_surfacemap",
                "filtered_reference_heatmap",
                "comparison_overview",
                "mark_ref_filtered_moved_surfacemap",
                "mark_ref_filtered_bb_surfacemap",
                "mark_comp_filtered_bb_surfacemap",
                "mark_comp_filtered_all_bb_surfacemap",
                "cell_accf_distribution",
            },
            id=ComparisonResponseImpression.__name__,
        ),
        pytest.param(LRResponseURL, {"lr_overview_plot"}, id=LRResponseURL.__name__),
    ],
)
@pytest.mark.parametrize("method", ["get_files", "generate_urls"])
class TestBaseResponseUrlsChildren:
    def test_expected_response_fields(
        self, klass: type[URLContainer], fields: set[str], method: str, tmp_path: Path
    ) -> None:
        """Every key returned by the method is present in the expected fields set."""
        func = getattr(klass, method)
        items = func(tmp_path) if method == "get_files" else func(_BASE_URL).model_dump(by_alias=True)

        missing = set(items) - fields
        assert not (missing := (set(items) - fields)), f"files {', '.join(missing)} are missing"
