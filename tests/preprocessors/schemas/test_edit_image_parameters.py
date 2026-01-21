from collections.abc import Callable
from itertools import chain
from typing import Any, Final

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from scipy.constants import micro

from preprocessors.schemas import EditImageParameters, Mask, RegressionOrder, Terms

DEFAULT_RESAMPLING_FACTOR: Final[int] = 4
DEFAULT_STEP_SIZE: Final[int] = 1
MASK: Final[Mask] = ((1, 0, 1), (0, 1, 0))  # type: ignore
CUTOFF_LENGTH: Final[float] = 250

type _EditImageParameter = Callable[..., EditImageParameters]


@pytest.fixture(scope="module")
def edit_image_parameter_(mask: Mask) -> _EditImageParameter:
    def wrapper(kwargs: dict[str, Any]):
        return EditImageParameters.model_validate({"mask": mask, "cutoff_length": CUTOFF_LENGTH} | kwargs)

    return wrapper


def get_error_fields(exc_info, typ: str) -> tuple[str, ...]:
    return tuple(chain.from_iterable(error["loc"] for error in exc_info.value.errors() if error["type"] == typ))


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
        assert params.cutoff_length == CUTOFF_LENGTH * micro
        assert params.step_size_x == DEFAULT_STEP_SIZE
        assert params.step_size_y == DEFAULT_STEP_SIZE
        assert params.crop is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"terms": Terms.PLANE},
            {"terms": Terms.SPHERE},
            {"regression_order": RegressionOrder.RO},
            {"regression_order": RegressionOrder.R1},
            {"regression_order": RegressionOrder.R2},
            {"crop": True},
        ],
    )
    def test_should_accept_valid_field_values(
        self, kwargs: dict[str, Any], edit_image_parameter_: _EditImageParameter
    ) -> None:
        """Test that enum and mask field values are accepted."""
        # Act
        params = edit_image_parameter_(kwargs)

        # Assert
        assert all(getattr(params, field) == value for field, value in kwargs.items())

    @given(valid_value=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("field", ["step_size_x", "step_size_y"])
    def test_should_accept_positive_integer_fields(
        self, field: str, valid_value: float, edit_image_parameter_: _EditImageParameter
    ) -> None:
        """Test that positive step sizes are accepted."""
        # Act
        params = edit_image_parameter_({field: valid_value})

        # Assert
        assert getattr(params, field) == valid_value

    @given(invalid_value=st.integers(max_value=0))
    @pytest.mark.parametrize("field", ["step_size_x", "step_size_y"])
    def test_should_reject_non_positive_integer_fields(
        self, field: str, invalid_value: float, edit_image_parameter_: _EditImageParameter
    ) -> None:
        """Test that step_size_x must be greater than 0."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            edit_image_parameter_({field: invalid_value})

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field,) for error in errors)

    @given(valid_value=st.floats(min_value=micro, max_value=3, allow_nan=False, allow_infinity=False))
    def test_should_accept_positive_cutoff_length(
        self, valid_value: float, edit_image_parameter_: _EditImageParameter
    ) -> None:
        """Test that positive step sizes are accepted."""
        # Act
        params = edit_image_parameter_({"cutoff_length": valid_value})

        # Assert
        assert params.cutoff_length == valid_value * micro

    @given(valid_value=st.floats(min_value=micro, max_value=3, allow_nan=False, allow_infinity=False))
    def test_should_accept_positive_resampling_factor(
        self, valid_value: float, edit_image_parameter_: _EditImageParameter
    ) -> None:
        """Test that positive step sizes are accepted."""
        # Act
        params = edit_image_parameter_({"resampling_factor": valid_value})

        # Assert
        assert params.resampling_factor == valid_value

    @given(invalid_value=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
    @pytest.mark.parametrize("field", ["cutoff_length", "resampling_factor"])
    def test_should_reject_non_positive_float_fields(
        self, field: str, invalid_value: float, edit_image_parameter_: _EditImageParameter
    ) -> None:
        """Test that step_size_x must be greater than 0."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            edit_image_parameter_({field: invalid_value})

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field,) for error in errors)

    def test_should_reject_when_required_fields_not_provided(self) -> None:
        """Test that validation fails when mask is not provided."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Field required") as exc_info:
            EditImageParameters()  # type: ignore

        # Assert
        assert get_error_fields(exc_info, "missing") == ("mask", "cutoff_length")
