import functools

from loguru import logger
from returns.io import IOFailure, IOSuccess
from returns.result import Failure, Success


def debug_function_signature(func):
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        logger.debug(f"{func.__name__}() returned {repr(value)}")
        return value

    return wrapper_debug


def log_io_railway_function(failure_message: str, success_message: str | None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            match result:
                case IOSuccess():
                    if success_message:
                        logger.info(success_message)
                case IOFailure(error):
                    logger.debug(f"{failure_message}: {error}")
                    logger.error(failure_message)

            return result

        return wrapper

    return decorator


def log_railway_function(failure_message: str, success_message: str | None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            match result:
                case Success():
                    if success_message:
                        logger.info(success_message)
                case Failure(error):
                    logger.debug(f"{failure_message}: {error}")
                    logger.error(failure_message)

            return result

        return wrapper

    return decorator
