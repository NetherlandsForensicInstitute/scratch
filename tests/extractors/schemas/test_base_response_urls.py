import inspect
from enum import StrEnum

import pytest
from pydantic import Field, HttpUrl, ValidationError

import extractors.schemas as schemas_module
from extractors.constants import ComparisonStriationFiles, LRFiles
from extractors.schemas import (
    ComparisonResponseStriation,
    ComparisonResponseStriationURL,
    LRResponse,
    LRResponseURL,
    SupportedExtension,
    URLContainer,
)

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
        return LRResponse(urls=LRResponseURL.from_enum(enum=LRFiles, base_url=_BASE_URL), lr=2.5, **_EMPTY_LR_STATS)

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
                lr="not-a-number",  # pyright: ignore[reportArgumentType]
                **_EMPTY_LR_STATS,
            )


class TestComparisonResponseStriation:
    """Tests for ComparisonResponseStriation serialization."""

    @pytest.fixture
    def striation_response(self) -> ComparisonResponseStriation:
        """ComparisonResponseStriation instance with valid URLs and comparison results."""
        return ComparisonResponseStriation(
            urls=ComparisonResponseStriationURL.from_enum(enum=ComparisonStriationFiles, base_url=_BASE_URL),
            comparison_results={
                "pixel_size": 1.5625e-6,
                "position_shift": 12.5e-6,
                "scale_factor": 1.0,
                "correlation_coefficient": 0.85,
                "overlap_length": 160e-6,
                "overlap_ratio": 0.804,
                "sa_ref": 0.19e-6,
                "sq_ref": 0.2395e-6,
                "sa_comp": 0.60e-6,
                "sq_comp": 0.7121e-6,
                "sa_diff": 0.50e-6,
                "sq_diff": 0.6138e-6,
                "ds_normalized_ref": 6.57,
                "ds_normalized_comp": 0.74,
                "ds_normalized_combined": 2.21,
                "shift_samples": 8,
                "overlap_samples": 102,
                "idx_reference_start": 8,
                "idx_compared_start": 0,
                "len_reference_equalized": 110,
                "len_compared_equalized": 102,
                "pixel_size_reference": 1.5625e-6,
                "pixel_size_compared": 1.5625e-6,
                "len_reference_original": 110,
                "len_compared_original": 102,
                "alignment_parameters": {
                    "max_scaling": 0.05,
                    "n_scale_steps": 7,
                    "min_overlap_distance": 350e-6,
                },
            },
        )

    def test_serialized_output_is_flat(self, striation_response: ComparisonResponseStriation) -> None:
        """model_dump produces a flat dict with no nested 'urls' key."""
        data = striation_response.model_dump()

        assert "urls" not in data

    def test_url_fields_promoted_to_top_level(self, striation_response: ComparisonResponseStriation) -> None:
        """All fields from the URL model appear at the top level."""
        url_fields = set(ComparisonResponseStriationURL.model_fields)
        data = striation_response.model_dump()

        assert url_fields <= set(data)

    def test_comparison_results_contains_expected_keys(self, striation_response: ComparisonResponseStriation) -> None:
        """Serialized output includes comparison_results with all expected metric keys."""
        expected_keys = {
            "pixel_size",
            "position_shift",
            "scale_factor",
            "correlation_coefficient",
            "overlap_length",
            "overlap_ratio",
            "sa_ref",
            "sq_ref",
            "sa_comp",
            "sq_comp",
            "sa_diff",
            "sq_diff",
            "ds_normalized_ref",
            "ds_normalized_comp",
            "ds_normalized_combined",
            "shift_samples",
            "overlap_samples",
            "idx_reference_start",
            "idx_compared_start",
            "len_reference_equalized",
            "len_compared_equalized",
            "pixel_size_reference",
            "pixel_size_compared",
            "len_reference_original",
            "len_compared_original",
            "alignment_parameters",
        }
        results = striation_response.model_dump()["comparison_results"]

        assert isinstance(results, dict)
        assert expected_keys <= set(results.keys()), f"Missing keys: {expected_keys - set(results.keys())}"

    def test_default_comparison_results_is_empty_dict(self) -> None:
        """comparison_results defaults to empty dict when not provided."""
        response = ComparisonResponseStriation(
            urls=ComparisonResponseStriationURL.from_enum(enum=ComparisonStriationFiles, base_url=_BASE_URL),
        )

        assert response.comparison_results == {}
