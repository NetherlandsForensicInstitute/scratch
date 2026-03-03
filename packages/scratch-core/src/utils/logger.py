import logging
from collections.abc import Callable
from enum import Enum
from functools import wraps
from itertools import chain
from typing import Any, Final

from loguru import logger
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Result, Success

VERBOSE: Final[bool] = False


def _debug_function_signature(func: Callable[..., Any], *args, **kwargs):
    """Print the function signature and return value."""
    signature = ", ".join(
        chain(
            (repr(arg) for arg in args),
            (f"{key}={repr(value)}" for key, value in kwargs.items()),
        )
    )
    logger.debug(f"Calling {func.__name__}({signature})")


class FailureLevel(Enum):
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def log_failure(failure_message: str, failure_level: FailureLevel, error: str) -> None:
    """
    Log a failure message at the specified severity level.

    :param failure_message: Human-readable failure description.
    :param failure_level: Severity level to log at (warning, error, or critical).
    :param error: Detailed error string, logged at debug level.
    """
    logger.debug(f"{failure_message}: {error}")
    match failure_level:
        case FailureLevel.WARNING:
            logger.warning(failure_message)
        case FailureLevel.ERROR:
            logger.error(failure_message)
        case FailureLevel.CRITICAL:
            logger.critical(failure_message)


def _log_io_container(
    result: IOResult,
    failure_message: str,
    success_message: str | None,
    failure_level: FailureLevel,
) -> None:
    match result:
        case IOSuccess():
            if success_message:
                logger.info(success_message)
        case IOFailure(error):
            log_failure(failure_message, failure_level, str(error))


def _log_container(
    result: Result,
    failure_message: str,
    success_message: str | None,
    failure_level: FailureLevel,
) -> None:
    match result:
        case Success():
            if success_message:
                logger.info(success_message)
        case Failure(error):
            log_failure(failure_message, failure_level, str(error))


def log_railway_function(
    failure_message: str,
    success_message: str | None = None,
    failure_level: FailureLevel = FailureLevel.ERROR,
):
    """
    Decorate a railway-oriented function to log its success or failure.

    Inspects the return value and logs based on whether the Result
    or IOResult container holds a success or failure value.

    :param failure_message: Message to log when the result is a failure.
    :param success_message: Message to log on success, or None to skip.
    :param failure_level: Logging level for failure messages.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if VERBOSE:
                _debug_function_signature(func, *args, **kwargs)
            result = func(*args, **kwargs)
            match result:
                case IOResult():
                    _log_io_container(result, failure_message, success_message, failure_level)
                case Result():
                    _log_container(result, failure_message, success_message, failure_level)
            return result

        return wrapper

    return decorator
