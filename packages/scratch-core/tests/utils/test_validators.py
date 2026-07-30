"""Unit tests for validate_enum_string function."""

from enum import IntEnum, StrEnum, auto
from typing import Annotated

import pytest
from pydantic import BaseModel, ValidationError

from utils.validators import validate_enum_string


class TestStrEnum(StrEnum):
    """Test StrEnum for unit tests."""

    OPTION_ONE = auto()
    OPTION_TWO = auto()
    OPTION_THREE = auto()


class TestIntEnum(IntEnum):
    """Test IntEnum for unit tests (mirrors SurfaceTerms)."""

    NONE = 0
    PLANE = 1
    SPHERE = 2


class TestStrModel(BaseModel):
    """Test pydantic model using StrEnum."""

    value: Annotated[TestStrEnum, validate_enum_string(TestStrEnum)]


class TestIntModel(BaseModel):
    """Test pydantic model using IntEnum."""

    value: Annotated[TestIntEnum, validate_enum_string(TestIntEnum)]


class TestValidateEnumString:
    """Tests for validate_enum_string function."""

    # -- StrEnum tests --

    def test_str_enum_valid_uppercase_string(self):
        """Test that uppercase string is converted to StrEnum."""
        result = TestStrModel.model_validate(dict(value="OPTION_ONE"))
        assert result.value == TestStrEnum.OPTION_ONE

    def test_str_enum_valid_lowercase_string(self):
        """Test that lowercase string is converted to StrEnum."""
        result = TestStrModel.model_validate(dict(value="option_two"))
        assert result.value == TestStrEnum.OPTION_TWO

    def test_str_enum_valid_mixed_case_string(self):
        """Test that mixed case string is converted to StrEnum."""
        result = TestStrModel.model_validate(dict(value="OpTiOn_ThReE"))
        assert result.value == TestStrEnum.OPTION_THREE

    def test_str_enum_instance_passes_through(self):
        """Test that passing an enum instance directly works."""
        result = TestStrModel(value=TestStrEnum.OPTION_ONE)
        assert result.value == TestStrEnum.OPTION_ONE
        result = TestStrModel.model_validate(dict(value=TestStrEnum.OPTION_ONE))
        assert result.value == TestStrEnum.OPTION_ONE

    def test_str_enum_invalid_value_raises_error(self):
        """Test that invalid value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TestStrModel.model_validate(dict(value="INVALID_OPTION"))

        error_msg = str(exc_info.value)
        assert "Invalid TestStrEnum" in error_msg
        assert "INVALID_OPTION" in error_msg

    def test_str_enum_error_message_includes_valid_options(self):
        """Test that error message lists valid enum members."""
        with pytest.raises(ValidationError) as exc_info:
            TestStrModel.model_validate(dict(value="WRONG"))

        error_msg = str(exc_info.value)
        assert "OPTION_ONE" in error_msg
        assert "OPTION_TWO" in error_msg
        assert "OPTION_THREE" in error_msg

    # -- IntEnum tests --

    def test_int_enum_valid_uppercase_string(self):
        """Test that uppercase string is converted to IntEnum."""
        result = TestIntModel.model_validate(dict(value="PLANE"))
        assert result.value == TestIntEnum.PLANE

    def test_int_enum_valid_lowercase_string(self):
        """Test that lowercase string is converted to IntEnum."""
        result = TestIntModel.model_validate(dict(value="sphere"))
        assert result.value == TestIntEnum.SPHERE

    def test_int_enum_valid_mixed_case_string(self):
        """Test that mixed case string is converted to IntEnum."""
        result = TestIntModel.model_validate(dict(value="None"))
        assert result.value == TestIntEnum.NONE

    def test_int_enum_valid_integer_value(self):
        """Test that integer value is converted to IntEnum."""
        result = TestIntModel.model_validate(dict(value=1))
        assert result.value == TestIntEnum.PLANE
        result = TestIntModel.model_validate(dict(value=2))
        assert result.value == TestIntEnum.SPHERE
        result = TestIntModel.model_validate(dict(value=0))
        assert result.value == TestIntEnum.NONE

    def test_int_enum_instance_passes_through(self):
        """Test that passing an IntEnum instance directly works."""
        result = TestIntModel(value=TestIntEnum.PLANE)
        assert result.value == TestIntEnum.PLANE
        result = TestIntModel.model_validate(dict(value=TestIntEnum.PLANE))
        assert result.value == TestIntEnum.PLANE

    def test_int_enum_invalid_string_raises_error(self):
        """Test that invalid string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TestIntModel.model_validate(dict(value="INVALID"))

        error_msg = str(exc_info.value)
        assert "Invalid TestIntEnum" in error_msg
        assert "INVALID" in error_msg

    def test_int_enum_invalid_integer_raises_error(self):
        """Test that invalid integer raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TestIntModel.model_validate(dict(value=99))

        error_msg = str(exc_info.value)
        assert "Invalid TestIntEnum" in error_msg
        assert "99" in error_msg
