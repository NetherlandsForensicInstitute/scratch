from collections.abc import Set
import logging
from typing import Callable, Final
import pytest
from unittest.mock import patch
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Result, Success

from utils.logger import log_railway_function


SUCCESS_MESSAGE: Final[str] = "Operation succeeded"
FAILURE_MESSAGE: Final[str] = "Operation failed"
ERROR_VALUE: Final[Exception] = RuntimeError("Something went wrong")


@log_railway_function(failure_message=FAILURE_MESSAGE, success_message=SUCCESS_MESSAGE)
def some_io_function(should_succeed: bool):
    if should_succeed:
        return IOSuccess(42)
    return IOFailure(ERROR_VALUE)


@log_railway_function(failure_message=FAILURE_MESSAGE, success_message=SUCCESS_MESSAGE)
def some_function(should_succeed: bool):
    if should_succeed:
        return Success(42)
    return Failure(ERROR_VALUE)


@log_railway_function(failure_message=FAILURE_MESSAGE, success_message=SUCCESS_MESSAGE)
def some_complex_function(a, *, b, c=3):
    """My function docstring."""
    return Success({"a": a, "x": [b, c]})


@pytest.mark.parametrize(
    "function, should_succeed, message, level",
    (
        pytest.param(some_function, True, SUCCESS_MESSAGE, {"INFO"}, id="Success"),
        pytest.param(
            some_function, False, FAILURE_MESSAGE, {"DEBUG", "ERROR"}, id="Failure"
        ),
        pytest.param(some_io_function, True, SUCCESS_MESSAGE, {"INFO"}, id="IOSuccess"),
        pytest.param(
            some_io_function, False, FAILURE_MESSAGE, {"DEBUG", "ERROR"}, id="IOFailure"
        ),
    ),
)
def test_log_railway_function_capture_log_message(
    function: Callable[[bool], Result | IOResult],
    should_succeed: bool,
    message: str,
    level: Set[str],
    caplog: pytest.LogCaptureFixture,
):
    """Test that IOSuccess logs an info message."""
    with caplog.at_level(logging.DEBUG):
        _ = function(should_succeed)

    assert message in caplog.text
    # Verify only INFO level is logged (not ERROR or DEBUG)
    assert {record.levelname for record in caplog.records} == level


@pytest.mark.parametrize("success_message", (None, ""))
def test_empty_success_message_does_not_log_on_success(
    success_message: str | None, caplog: pytest.LogCaptureFixture
):
    """Test that None/empty success_message doesn't log info on success."""

    @log_railway_function(
        failure_message=FAILURE_MESSAGE, success_message=success_message
    )
    def empty_success_message_func():
        return IOSuccess(100)

    with caplog.at_level(logging.DEBUG):
        _ = empty_success_message_func()

    assert not {record.levelname for record in caplog.records}


def test_decorator_is_not_destructive():
    """Test that decorator does not modify function's result."""
    result = some_complex_function(1, b=2, c=5)

    assert isinstance(result, Success)
    assert result.unwrap() == {"a": 1, "x": [2, 5]}


def test_decorator_preserves_function_metadata():
    """Test decorator preserves function metadata."""
    assert some_complex_function.__name__ == "some_complex_function"
    assert some_complex_function.__doc__ == "My function docstring."


@patch("utils.logger.VERBOSE", True)
def test_verbose_mode_logs_function_signature(caplog: pytest.LogCaptureFixture):
    """Test that VERBOSE mode logs the function signature."""

    with caplog.at_level(logging.DEBUG):
        _ = some_complex_function(1, b=2, c=5)

    assert "Calling some_complex_function(1, b=2, c=5)" in caplog.text
