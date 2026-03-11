import inspect
from pathlib import Path

import pytest
from pydantic import Field, HttpUrl, ValidationError

import extractors.schemas as schemas_module
from extractors.schemas import (
    BaseResponseURLs,
    ComparisonResponseImpression,
    ComparisonResponseStriation,
    ComparisonResponseStriationURL,
    LRResponse,
    LRResponseURL,
    PrepareMarkResponseImpression,
    PrepareMarkResponseStriation,
    ProcessedDataAccess,
    SupportedExtension,
)

_CONCRETE_CLASSES = [
    cls
    for _, cls in inspect.getmembers(schemas_module, inspect.isclass)
    if cls.__module__ == schemas_module.__name__
    and cls is not BaseResponseURLs
    and cls is not SupportedExtension
    and cls is not LRResponse
]

_BASE_URL = "http://localhost:8000/files/token123"


@pytest.mark.parametrize("cls", _CONCRETE_CLASSES, ids=lambda cls: cls.__name__)
def test_all_module_classes_extend_base_response_urls(cls: type[BaseResponseURLs]) -> None:
    """Every model class in the schemas module inherits from BaseResponseURLs."""
    assert issubclass(cls, BaseResponseURLs)


class _SimpleURLs(BaseResponseURLs):
    """Minimal concrete subclass with one aliased and one plain field."""

    aliased_file: HttpUrl = Field(
        ...,
        alias="my_alias",
        json_schema_extra={"file_name": "aliased.png"},
    )
    plain_file: HttpUrl = Field(
        ...,
        json_schema_extra={"file_name": "plain.png"},
    )


class _NoFileName(BaseResponseURLs):
    """Subclass with a field whose json_schema_extra is missing 'file_name'."""

    broken_field: HttpUrl = Field(..., json_schema_extra={})


class TestGetFiles:
    """Tests for BaseResponseURLs.get_files."""

    @pytest.mark.parametrize(
        ("field", "alias"),
        [
            pytest.param("aliased_file", "my_alias", id="aliased_field"),
            pytest.param("plain_file", "plain_file", id="non_aliased_field"),
        ],
    )
    def test_key_and_path_for_field(self, tmp_path: Path, field: str, alias: str) -> None:
        """Fields are mapped to resource_directory / file_name."""
        filename = _SimpleURLs.model_fields[field].json_schema_extra["file_name"]  # type: ignore
        files = _SimpleURLs.get_files(tmp_path)

        assert alias in files
        assert filename
        assert files[alias] == tmp_path / str(filename)

    def test_all_fields_are_represented(self, tmp_path: Path) -> None:
        """The returned dict contains one entry per model field."""
        fields = set(_SimpleURLs.model_json_schema(by_alias=True)["properties"])
        files = _SimpleURLs.get_files(tmp_path)

        assert not (missing := (set(files) - fields)), f"files {', '.join(missing)} are missing"

    def test_raises_key_error_when_file_name_missing_from_json_schema_extra(self, tmp_path: Path) -> None:
        """get_files raises KeyError when a field's json_schema_extra lacks 'file_name'."""
        with pytest.raises(KeyError, match="file_name"):
            _NoFileName.get_files(tmp_path)


class TestGenerateUrls:
    """Tests for BaseResponseURLs.generate_urls."""

    def test_aliased_field_populated_via_alias(self) -> None:
        """Aliased fields are set using their alias key."""
        result = _SimpleURLs.generate_urls(_BASE_URL)
        assert result.aliased_file == HttpUrl(f"{_BASE_URL}/aliased.png")
        assert result.plain_file == HttpUrl(f"{_BASE_URL}/plain.png")

    @pytest.mark.parametrize("protocol", ["http", "https"])
    def test_supports_http_and_https(self, protocol: str) -> None:
        """Both HTTP and HTTPS base URLs are accepted."""
        base_url = f"{protocol}://localhost:8000/files/token"
        result = _SimpleURLs.generate_urls(base_url)

        assert str(result.aliased_file).startswith(protocol)

    def test_all_fields_are_represented(self) -> None:
        """The returned dict contains one entry per model field."""
        fields = set(_SimpleURLs.model_json_schema(by_alias=True)["properties"])
        urls = _SimpleURLs.generate_urls(_BASE_URL).model_dump(by_alias=True)

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
            _SimpleURLs.generate_urls(invalid_url)


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


class TestComparisonResponseStriation:
    """Tests for ComparisonResponseStriation serialization."""

    @pytest.fixture
    def striation_response(self) -> ComparisonResponseStriation:
        """ComparisonResponseStriation instance with valid URLs and comparison results."""
        return ComparisonResponseStriation(
            urls=ComparisonResponseStriationURL.generate_urls(_BASE_URL),
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
            urls=ComparisonResponseStriationURL.generate_urls(_BASE_URL),
        )

        assert response.comparison_results == {}


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
            ComparisonResponseStriationURL,
            {
                "mark_ref_surfacemap",
                "mark_comp_surfacemap",
                "filtered_reference_heatmap",
                "comparison_overview",
                "mark_ref_preview",
                "mark_comp_preview",
                "similarity_plot",
                "filtered_compared_heatmap",
                "side_by_side_heatmap",
                "wavedlength_plot",
                "mark_reference_aligned_meta",
                "mark_reference_aligned_data",
                "mark_compared_aligned_data",
                "mark_compared_aligned_meta",
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
        self, klass: type[BaseResponseURLs], fields: set[str], method: str, tmp_path: Path
    ) -> None:
        """Every key returned by the method is present in the expected fields set."""
        func = getattr(klass, method)
        items = func(tmp_path) if method == "get_files" else func(_BASE_URL).model_dump(by_alias=True)

        assert not (missing := (set(items) - fields)), f"files {', '.join(missing)} are missing"
