from functools import wraps
from typing import Any, Callable, Final
from itertools import chain

from loguru import logger
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Result, Success

VERBOSE: Final[bool] = False


def _debug_function_signature(func: Callable[..., Any], *args, **kwargs):
    """Print the function signature and return value"""
    signature = ", ".join(
        chain(
            (repr(arg) for arg in args),
            (f"{key}={repr(value)}" for key, value in kwargs.items()),
        )
    )
    logger.debug(f"Calling {func.__name__}({signature})")


def _log_io_container(
    result: IOResult, failure_message: str, success_message: str | None
) -> None:
    match result:
        case IOSuccess():
            if success_message:
                logger.info(success_message)
        case IOFailure(error):
            logger.debug(f"{failure_message}: {error}")
            logger.error(failure_message)


def _log_container(
    result: Result, failure_message: str, success_message: str | None
) -> None:
    match result:
        case Success():
            if success_message:
                logger.info(success_message)
        case Failure(error):
            logger.debug(f"{failure_message}: {error}")
            logger.error(failure_message)


def log_railway_function(failure_message: str, success_message: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if VERBOSE:
                _debug_function_signature(func, *args, **kwargs)
            result = func(*args, **kwargs)
            match result:
                case IOResult():
                    _log_io_container(result, failure_message, success_message)
                case Result():
                    _log_container(result, failure_message, success_message)
            return result

        return wrapper

    return decorator
