from enum import Enum, IntEnum
from typing import Any, TypeVar

from pydantic import BeforeValidator


E = TypeVar("E", bound=Enum)


def validate_enum_string(enum_class: type[E]) -> Any:
    """
    Create a BeforeValidator that converts string/int values into enum members.

    :param enum_class: The enum class to validate against
    :returns: BeforeValidator callable
    """

    def validator(value: str | int | E) -> E:
        if isinstance(value, enum_class):
            return value

        # Handle IntEnum with integer values
        if isinstance(enum_class, type) and issubclass(enum_class, IntEnum):
            if isinstance(value, int):
                try:
                    return enum_class(value)
                except ValueError:
                    pass

        value_str = str(value).upper()
        if value_str not in enum_class.__members__:
            raise ValueError(
                f"Invalid {enum_class.__name__}: '{value}'. "
                f"Must be one of {list(enum_class.__members__.keys())}"
            )
        return enum_class[value_str]

    return BeforeValidator(validator)
