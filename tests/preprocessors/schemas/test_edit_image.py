from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pydantic import ValidationError

from constants import PROJECT_ROOT
from preprocessors.schemas import EditImage, Filter, Level


@pytest.fixture(scope="module")
def valid_parsed_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    parsed_file = tmp_path_factory.mktemp("edit_image_tmp") / "test.x3p"
    parsed_file.touch()
    return parsed_file


def test_edit_image_fails_no_task_field_given(valid_parsed_file: Path) -> None:
    """Test EditImage fails when no task param is given (default values)."""
    # Arrange & Assert
    with pytest.raises(ValidationError, match="No edit task parmeters given"):
        _ = EditImage(parsed_file=valid_parsed_file)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("level", Level.PLAIN, id="level"),
        pytest.param("filter", Filter.R1, id="filter"),
        pytest.param("zoom", True, id="zoom"),
        # mask_array already covered by `test_mask_array_with_valid_boundary_values`
    ],
)
def test_edit_image_at_least_one_task_field_given(
    valid_parsed_file: Path, field: str, value: Level | Filter | bool
) -> None:
    """Test EditImage validates successfully when at least one task field is provided."""
    # Act
    edit_image = EditImage.model_validate({"parsed_file": valid_parsed_file, field: value})

    # Assert
    assert getattr(edit_image, field) == value


@given(
    mask_array=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=50),
            st.integers(min_value=1, max_value=50),
        ),
        elements=st.sampled_from([0.0, 255.0]),
    )
)
def test_mask_array_with_valid_boundary_values(mask_array: np.ndarray, valid_parsed_file: Path) -> None:
    """Test mask_array validator with boundary values (0 and 255) using property-based testing."""
    # Act
    edit_image = EditImage(parsed_file=valid_parsed_file, mask_array=mask_array)

    # Assert
    assert np.array_equal(edit_image.mask_array, mask_array)


@given(
    invalid_value=st.one_of(
        st.floats(max_value=-0.1, allow_nan=False, allow_infinity=False),
        st.floats(min_value=255.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
)
def test_mask_array_with_invalid_values_raises_error(invalid_value: float, valid_parsed_file: Path) -> None:
    """Test that mask_array with values outside [0, 255] raises ValidationError."""
    # Arrange
    mask_array = np.array([[invalid_value, 100.0, 200.0], [50.0, 100.0, 200.0]], dtype=np.float64)

    # Act & Assert
    with pytest.raises(ValidationError, match="mask_array values must be between 0 and 255"):
        EditImage(parsed_file=valid_parsed_file, mask_array=mask_array)


@given(
    mask_array=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.integers(min_value=1, max_value=100),
        ),
        elements=st.floats(min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False),
    )
)
def test_mask_array_with_valid_float64_values(mask_array: np.ndarray, valid_parsed_file: Path) -> None:
    """Test that any float64 array with values 0-255 is valid using property-based testing."""
    # Act
    edit_image = EditImage(parsed_file=valid_parsed_file, mask_array=mask_array)

    # Assert
    assert np.array_equal(edit_image.mask_array, mask_array)


@pytest.mark.parametrize(
    ("path", "expected_message"),
    [
        pytest.param(Path("/nonexistent/path/parsed.x3p"), "Path does not point to a file", id="non-existent file"),
        pytest.param(Path.cwd(), "Path does not point to a file", id="directory"),
        pytest.param(PROJECT_ROOT / "pyproject.toml", "was expecting an x3p file: pyproject.toml", id="non-x3p file"),
    ],
)
def test_nonexistent_parsed_file_raises_error(path: Path, expected_message: str) -> None:
    """Test that non-existent parsed file raises ValidationError."""
    # Act & Assert
    with pytest.raises(ValidationError, match=expected_message):
        EditImage(parsed_file=path)


@given(non_positive_value=st.integers(max_value=0))
def test_sampling_must_be_positive(non_positive_value: int, valid_parsed_file: Path) -> None:
    """Test that sampling must be a positive integer (rejects 0 and negative values)."""
    # Act & Assert
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        EditImage(parsed_file=valid_parsed_file, sampling=non_positive_value)


@pytest.mark.parametrize(
    ("key", "values"),
    [
        pytest.param("level", (Level.PLAIN, Level.SPHERE), id="level"),
        pytest.param("filter", (Filter.RO, Filter.R1, Filter.R2), id="filter"),
    ],
)
def test_all_enum_values(valid_parsed_file: Path, key: str, values: tuple[Enum, ...]) -> None:
    """Test that all Level enum values are accepted."""
    # Act and assert
    assert all(
        getattr(EditImage.model_validate({"parsed_file": valid_parsed_file, key: value}), key) == value
        for value in values
    )
