"""Unit tests for validate_enum_string function."""

from enum import StrEnum, auto

import pytest
from pydantic import BaseModel, ValidationError
from typing import Annotated

from conversion.export.validators import validate_enum_string


class TestEnum(StrEnum):
    """Test enum for unit tests."""

    OPTION_ONE = auto()
    OPTION_TWO = auto()
    OPTION_THREE = auto()


class TestModel(BaseModel):
    """Test pydantic model for unit tests."""

    value: Annotated[TestEnum, validate_enum_string(TestEnum)]


class TestValidateEnumString:
    """Tests for validate_enum_string function."""

    def test_valid_uppercase_string(self):
        """Test that uppercase string is converted to enum."""
        result = TestModel.model_validate(dict(value="OPTION_ONE"))
        assert result.value == TestEnum.OPTION_ONE

    def test_valid_lowercase_string(self):
        """Test that lowercase string is converted to enum."""
        result = TestModel.model_validate(dict(value="option_two"))
        assert result.value == TestEnum.OPTION_TWO

    def test_valid_mixed_case_string(self):
        """Test that mixed case string is converted to enum."""
        result = TestModel.model_validate(dict(value="OpTiOn_ThReE"))
        assert result.value == TestEnum.OPTION_THREE

    def test_enum_instance_passes_through(self):
        """Test that passing an enum instance directly works."""
        result = TestModel(value=TestEnum.OPTION_ONE)
        assert result.value == TestEnum.OPTION_ONE
        result = TestModel.model_validate(dict(value=TestEnum.OPTION_ONE))
        assert result.value == TestEnum.OPTION_ONE

    def test_invalid_enum_value_raises_error(self):
        """Test that invalid enum value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TestModel.model_validate(dict(value="INVALID_OPTION"))

        error_msg = str(exc_info.value)
        assert "Invalid TestEnum" in error_msg
        assert "INVALID_OPTION" in error_msg

    def test_error_message_includes_valid_options(self):
        """Test that error message lists valid enum members."""
        with pytest.raises(ValidationError) as exc_info:
            TestModel.model_validate(dict(value="WRONG"))

        error_msg = str(exc_info.value)
        assert "OPTION_ONE" in error_msg
        assert "OPTION_TWO" in error_msg
        assert "OPTION_THREE" in error_msg
