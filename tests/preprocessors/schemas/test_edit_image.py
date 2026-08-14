from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Any

import pytest
from conversion.data_formats import SurfaceTerms
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from preprocessors.schemas import EditImage, RegressionOrder
from tests.constants import CUTOFF_LENGTH, DEFAULT_RESAMPLING_FACTOR


def get_error_fields(exc_info, typ: str) -> tuple[str, ...]:
    return tuple(chain.from_iterable(error["loc"] for error in exc_info.value.errors() if error["type"] == typ))


class TestEditImage:
    """Tests for EditImage request model."""

    def test_should_reject_non_x3p_file(
        self, scan_directory: Path, edit_image_parameter: Callable[..., EditImage]
    ) -> None:
        """Test that EditImage rejects files that are not X3P format."""
        # Arrange
        al3d_file = scan_directory / "circle.al3d"

        # Act & Assert
        with pytest.raises(ValidationError, match="Unsupported extension") as exc_info:
            edit_image_parameter(scan_file=al3d_file)

        # Assert
        errors = exc_info.value.errors()
        error_messages = [str(error["ctx"]["error"]) for error in errors if "ctx" in error]
        assert any("Unsupported extension" in msg for msg in error_messages)

    def test_should_reject_non_existent_file(
        self, tmp_path: Path, edit_image_parameter: Callable[..., EditImage]
    ) -> None:
        """Test that EditImage rejects non-existent files."""
        # Arrange
        non_existent_file = tmp_path / "does_not_exist.x3p"

        # Act & Assert
        with pytest.raises(ValidationError, match="Path does not point to a file") as exc_info:
            edit_image_parameter(scan_file=non_existent_file)

        # Assert
        errors = exc_info.value.errors()
        assert any("scan_file" in error["loc"] for error in errors)

    def test_should_create_with_all_defaults(self, edit_image_parameter: Callable[..., EditImage]) -> None:
        """Test that EditImage can be created with default values when mask and cutoff_length are provided."""
        # Arrange

        # Act
        params = edit_image_parameter()

        # Assert
        assert params.resampling_factor == DEFAULT_RESAMPLING_FACTOR
        assert params.surface_terms == SurfaceTerms.NONE
        assert params.regression_order == RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE
        assert params.cutoff_length == CUTOFF_LENGTH
        assert params.crop is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"surface_terms": SurfaceTerms.PLANE},
            {"surface_terms": SurfaceTerms.SPHERE},
            {"regression_order": RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE},
            {"regression_order": RegressionOrder.LOCAL_PLANAR},
            {"regression_order": RegressionOrder.LOCAL_QUADRATIC},
            {"crop": True},
        ],
    )
    def test_should_accept_valid_field_values(
        self, kwargs: dict[str, Any], edit_image_parameter: Callable[..., EditImage]
    ) -> None:
        """Test that enum and mask field values are accepted."""
        # Act
        params = edit_image_parameter(**kwargs)

        # Assert
        assert all(getattr(params, field) == value for field, value in kwargs.items())

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            ("plane", SurfaceTerms.PLANE),
            ("PLANE", SurfaceTerms.PLANE),
            ("Plane", SurfaceTerms.PLANE),
            ("sphere", SurfaceTerms.SPHERE),
            ("SPHERE", SurfaceTerms.SPHERE),
            ("none", SurfaceTerms.NONE),
            ("NONE", SurfaceTerms.NONE),
            (1, SurfaceTerms.PLANE),
            (2, SurfaceTerms.SPHERE),
            (0, SurfaceTerms.NONE),
        ],
    )
    def test_should_accept_string_and_int_for_surface_terms(
        self,
        input_value: str | int,
        expected: SurfaceTerms,
        edit_image_parameter: Callable[..., EditImage],
    ) -> None:
        """Test that surface_terms accepts both string and int values (non-breaking API)."""
        # Act
        params = edit_image_parameter(surface_terms=input_value)

        # Assert
        assert params.surface_terms == expected

    @given(valid_value=st.floats(min_value=1e-6, max_value=250e-6, allow_nan=False, allow_infinity=False))
    def test_should_accept_positive_cutoff_length(
        self, valid_value: float, edit_image_parameter: Callable[..., EditImage]
    ) -> None:
        """Test that cutoff_length values within the plausible range (1e-6-250e-6 m) are accepted."""
        # Act
        params = edit_image_parameter(cutoff_length=valid_value)

        # Assert
        assert params.cutoff_length == valid_value

    @given(valid_value=st.floats(min_value=1, max_value=3, allow_nan=False, allow_infinity=False))
    def test_should_accept_positive_resampling_factor(
        self, valid_value: float, edit_image_parameter: Callable[..., EditImage]
    ) -> None:
        """Test that positive resampling factor are accepted."""
        # Act
        params = edit_image_parameter(resampling_factor=valid_value)

        # Assert
        assert params.resampling_factor == valid_value

    @given(invalid_value=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
    @pytest.mark.parametrize("field", ["cutoff_length", "resampling_factor"])
    def test_should_reject_non_positive_float_fields(
        self, field: str, invalid_value: float, edit_image_parameter: Callable[..., EditImage]
    ) -> None:
        """Test that step_size_x must be greater than 0."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            edit_image_parameter(**{field: invalid_value})

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field,) for error in errors)

    def test_should_reject_when_required_fields_not_provided(self) -> None:
        """Test that validation fails when mask is not provided."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Field required") as exc_info:
            EditImage()  # type: ignore

        # Assert
        assert get_error_fields(exc_info, "missing") == ("scan_file", "cutoff_length", "surface_terms")
