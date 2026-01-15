from collections.abc import Callable
from itertools import chain
from typing import Any, Final

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from scipy.constants import micro

from preprocessors.schemas import EditImageParameters, Mask, RegressionOrder, Terms

DEFAULT_RESAMPLING_FACTOR: Final[int] = 4
DEFAULT_STEP_SIZE: Final[int] = 1
CUTOFF_LENGTH: Final[float] = 250 * micro


@pytest.fixture(scope="module")
def composable_edit_image_parameter(mask: Mask) -> Callable[..., EditImageParameters]:
    def wrapper(**kwargs) -> EditImageParameters:
        return EditImageParameters(mask=mask, cutoff_length=CUTOFF_LENGTH, **kwargs)

    return wrapper


class TestEditImageParameters:
    """Tests for EditImageParameters configuration model."""

    def test_should_create_with_all_defaults(self, mask: Mask) -> None:
        """Test that EditImageParameters can be created with default values when mask and cutoff_length are provided."""
        # Arrange

        # Act
        params = EditImageParameters(mask=mask, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Assert
        assert params.resampling_factor == DEFAULT_RESAMPLING_FACTOR
        assert params.terms == Terms.PLANE
        assert params.regression_order == RegressionOrder.RO
        np.testing.assert_array_equal(params.mask.mask_array, mask.mask_array)
        assert params.cutoff_length == CUTOFF_LENGTH
        assert params.step_size_x == DEFAULT_STEP_SIZE
        assert params.step_size_y == DEFAULT_STEP_SIZE
        assert params.overwrite is False
        assert params.crop is False

    @given(invalid_factor=st.integers(max_value=0))
    def test_should_reject_non_positive_resampling_factor(
        self, invalid_factor: int, composable_edit_image_parameter: Callable[..., EditImageParameters]
    ) -> None:
        """Test that resampling_factor must be positive."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            composable_edit_image_parameter(resampling_factor=invalid_factor)

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("resampling_factor",) for error in errors)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"terms": Terms.PLANE},
            {"terms": Terms.SPHERE},
            {"regression_order": RegressionOrder.RO},
            {"regression_order": RegressionOrder.R1},
            {"regression_order": RegressionOrder.R2},
            {"overwrite": True},
            {"crop": True},
        ],
    )
    def test_should_accept_valid_field_values(
        self, kwargs: dict[str, Any], composable_edit_image_parameter: Callable[..., EditImageParameters]
    ) -> None:
        """Test that enum and mask field values are accepted."""
        # Act
        params = composable_edit_image_parameter(**kwargs)

        # Assert
        assert all(getattr(params, field) == value for field, value in kwargs.items())

    @given(step_size=st.integers(min_value=1, max_value=1000))
    def test_should_accept_positive_step_sizes(
        self, step_size: int, composable_edit_image_parameter: Callable[..., EditImageParameters]
    ) -> None:
        """Test that positive step sizes are accepted."""
        # Act
        params = composable_edit_image_parameter(step_size_x=step_size, step_size_y=step_size)

        # Assert
        assert params.step_size_x == params.step_size_y == step_size

    @given(invalid_step=st.integers(max_value=0))
    @pytest.mark.parametrize("field", ["step_size_x", "step_size_y"])
    def test_should_reject_non_positive_step_size(
        self, field: str, invalid_step: int, composable_edit_image_parameter: Callable[..., EditImageParameters]
    ) -> None:
        """Test that step_size_x must be greater than 0."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            composable_edit_image_parameter(**{field: invalid_step})

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field,) for error in errors)

    def test_should_reject_when_required_fields_not_provided(self) -> None:
        """Test that validation fails when mask is not provided."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Field required") as exc_info:
            EditImageParameters()  # type: ignore

        # Assert
        assert tuple(
            chain.from_iterable(error["loc"] for error in exc_info.value.errors() if error["type"] == "missing")
        ) == ("mask", "cutoff_length")

    @given(cutoff=st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_infinity=False))
    def test_should_accept_positive_cutoff_length(self, cutoff: float, mask: Mask) -> None:
        """Test that positive cutoff_length values are accepted."""
        # Act
        params = EditImageParameters(mask=mask, cutoff_length=cutoff)  # type: ignore

        # Assert
        assert params.cutoff_length == cutoff

    @given(invalid_cutoff=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False) | st.integers(max_value=0))
    def test_should_reject_non_positive_cutoff_length(self, invalid_cutoff: float, mask: Mask) -> None:
        """Test that cutoff_length must be positive."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0"):
            EditImageParameters(mask=mask, cutoff_length=invalid_cutoff)  # type: ignore
