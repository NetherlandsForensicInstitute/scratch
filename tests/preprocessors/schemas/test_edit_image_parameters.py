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
MASK: Final[Mask] = ((True, False), (False, True))


class TestEditImageParameters:
    """Tests for EditImageParameters configuration model."""

    def test_should_create_with_all_defaults(self) -> None:
        """Test that EditImageParameters can be created with default values when mask and cutoff_length are provided."""
        # Arrange

        # Act
        params = EditImageParameters(mask=MASK, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Assert
        assert params.resampling_factor == DEFAULT_RESAMPLING_FACTOR
        assert params.terms == Terms.PLANE
        assert params.regression_order == RegressionOrder.RO
        assert params.mask == MASK
        assert params.cutoff_length == CUTOFF_LENGTH
        assert params.step_size_x == DEFAULT_STEP_SIZE
        assert params.step_size_y == DEFAULT_STEP_SIZE
        assert params.overwrite is False
        assert params.crop is False

    @given(invalid_factor=st.integers(max_value=0))
    def test_should_reject_non_positive_resampling_factor(self, invalid_factor: int) -> None:
        """Test that resampling_factor must be positive."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            EditImageParameters(resampling_factor=invalid_factor, mask=MASK, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("resampling_factor",) for error in errors)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("terms", Terms.PLANE),
            ("terms", Terms.SPHERE),
            ("regression_order", RegressionOrder.RO),
            ("regression_order", RegressionOrder.R1),
            ("regression_order", RegressionOrder.R2),
            ("overwrite", True),
            ("crop", True),
        ],
    )
    def test_should_accept_valid_field_values(self, field: str, value: Any) -> None:
        """Test that enum and mask field values are accepted."""
        # Act
        params = EditImageParameters.model_validate({
            field: value,
            "mask": MASK,
            "cutoff_length": CUTOFF_LENGTH,
        })  # type: ignore

        # Assert
        assert getattr(params, field) == value

    @pytest.mark.parametrize(
        "invalid_mask",
        [
            pytest.param([True, False, True], id="1D array"),
            pytest.param([[[True, False], [False, True]]], id="3D array"),
            pytest.param([[True], []], id="mismatch shape"),
            pytest.param([[]], id="Empty nested list"),
            pytest.param([[], []], id="Empty 2D array"),
        ],
    )
    def test_should_reject_invalid_mask_dimensions(self, invalid_mask) -> None:
        """Test that mask validation rejects non-2D arrays and empty masks."""
        # Act & Assert
        with pytest.raises(ValidationError, match="mask") as exc_info:
            EditImageParameters(mask=invalid_mask, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any("mask" in str(error["loc"]) for error in errors)

    def test_should_coerce_numeric_values_to_boolean(self) -> None:
        """Test that numeric values (0, 1) are coerced to boolean by Pydantic."""
        # Arrange
        mask = ((1, 0, 1), (0, 1, 0))

        # Act
        params = EditImageParameters(mask=mask, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Assert
        assert params.mask == ((True, False, True), (False, True, False))

    @given(
        width=st.integers(min_value=2, max_value=100),
        data=st.data(),
    )
    def test_mask_array_should_return_numpy_array(self, width: int, data: st.DataObject) -> None:
        """Test that mask_array property returns a numpy array with property-based testing.

        The Mask type is defined as tuple[tuple[bool, ...], tuple[bool, ...]],
        which means exactly 2 rows. We generate masks with varying widths (2-100).
        """
        # Arrange - Generate mask with exactly 2 rows and consistent column lengths
        mask = (
            tuple(data.draw(st.booleans()) for _ in range(width)),
            tuple(data.draw(st.booleans()) for _ in range(width)),
        )
        params = EditImageParameters(mask=mask, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Act
        mask_array = params.mask_array

        # Assert
        assert isinstance(mask_array, np.ndarray)
        assert mask_array.dtype == np.bool_
        assert mask_array.shape == (2, width)

    def test_mask_array_should_have_correct_values(self) -> None:
        """Test that mask_array property preserves boolean values from mask."""
        # Arrange
        mask = ((1, 0), (0, 1))
        params = EditImageParameters(mask=mask, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Act
        mask_array = params.mask_array

        # Assert
        assert mask_array[0, 0] is np.True_
        assert mask_array[0, 1] is np.False_
        assert mask_array[1, 0] is np.False_
        assert mask_array[1, 1] is np.True_

    @given(step_size=st.integers(min_value=1, max_value=1000))
    def test_should_accept_positive_step_sizes(self, step_size: int) -> None:
        """Test that positive step sizes are accepted."""
        # Act
        params = EditImageParameters(
            step_size_x=step_size, step_size_y=step_size, mask=MASK, cutoff_length=CUTOFF_LENGTH
        )  # type: ignore

        # Assert
        assert params.step_size_x == step_size
        assert params.step_size_y == step_size

    @given(invalid_step=st.integers(max_value=0))
    @pytest.mark.parametrize("field", ["step_size_x", "step_size_y"])
    def test_should_reject_non_positive_step_size(self, field: str, invalid_step: int) -> None:
        """Test that step_size_x must be greater than 0."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            EditImageParameters.model_validate({
                field: invalid_step,
                "mask": MASK,
                "cutoff_length": CUTOFF_LENGTH,
            })  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field,) for error in errors)

    def test_should_reject_when_mask_not_provided(self) -> None:
        """Test that validation fails when mask is not provided."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Field required") as exc_info:
            EditImageParameters()  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("mask",) and error["type"] == "missing" for error in errors)

    def test_should_accept_when_both_mask_and_crop_provided(self) -> None:
        """Test that validation passes when both mask and crop are provided."""
        # Act
        params = EditImageParameters(mask=((True,), (False,)), crop=True, cutoff_length=CUTOFF_LENGTH)  # type: ignore

        # Assert
        assert params.mask is not None
        assert params.mask == ((True,), (False,))
        assert params.crop is True

    @given(cutoff=st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_infinity=False))
    def test_should_accept_positive_cutoff_length(self, cutoff: float) -> None:
        """Test that positive cutoff_length values are accepted."""
        # Act
        params = EditImageParameters(mask=MASK, cutoff_length=cutoff)  # type: ignore

        # Assert
        assert params.cutoff_length == cutoff

    @given(invalid_cutoff=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False) | st.integers(max_value=0))
    def test_should_reject_non_positive_cutoff_length(self, invalid_cutoff: float) -> None:
        """Test that cutoff_length must be positive."""
        # Act & Assert
        with pytest.raises(ValidationError, match="greater than 0") as exc_info:
            EditImageParameters(mask=MASK, cutoff_length=invalid_cutoff)  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("cutoff_length",) for error in errors)

    def test_should_reject_when_cutoff_length_not_provided(self) -> None:
        """Test that validation fails when cutoff_length is not provided."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Field required") as exc_info:
            EditImageParameters(mask=MASK)  # type: ignore

        # Assert
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("cutoff_length",) and error["type"] == "missing" for error in errors)
