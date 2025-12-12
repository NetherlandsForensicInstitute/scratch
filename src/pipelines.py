"""
Railway-oriented programming pipeline utilities.

This module provides a simplified interface for executing functional pipelines using
railway-oriented programming (ROP) patterns, abstracting away the complexity of working
directly with the returns library's container types (IOResultE, ResultE, etc.).

Railway-oriented programming is a functional error-handling pattern where operations
are chained together in a "railway" with two tracks: a success track and a failure track.
Each operation either continues on the success track or switches to the failure track,
propagating errors automatically without explicit error checking at each step.

The main entry point, `run_pipeline`, allows developers to compose multiple operations
without manually handling container unwrapping, monadic binding, or error propagation.
It automatically:
- Handles both raw values and Container types as input
- Binds operations together using monadic composition
- Unwraps the final result or raises an HTTPException on failure

This abstraction enables cleaner, more maintainable code by hiding the underlying
railway container mechanics while preserving type safety and functional error handling.
"""

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException
from returns.interfaces.container import ContainerN
from returns.io import IOResultE, IOSuccess
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import ResultE, Success


def _capture_ioresult_value[T](result: IOResultE[T] | ResultE[T], error_message: str) -> T:
    match result:
        case IOSuccess(Success(value)) | Success(value):
            return value
        case _:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=error_message)


def _pipeline_flow[T](entry_value: Any | ContainerN, *pipeline: Callable[..., Any]) -> IOResultE[T] | ResultE[T]:
    first_function = None
    pipeline_tasks: Any = pipeline
    if not isinstance(entry_value, ContainerN):
        first_function, *pipeline_tasks = pipeline

    return flow(
        entry_value,
        *((first_function,) if first_function else ()),
        *[bind(task) for task in pipeline_tasks],
    )


def run_pipeline(entry_value: Any | ContainerN, *tasks: Callable[[Any], Any], error_message: str) -> Any:
    """
    Execute a series of tasks in a functional pipeline and return the final result.

    This function orchestrates a railway-oriented programming pipeline using the
    ``returns`` library. It takes an entry value and a sequence of tasks, executes
    them in order using monadic binding, and unwraps the final result.

    :param entry_value: The initial value to pass into the pipeline. This may be a
        raw value or a Container from the ``returns`` library (e.g. ``IOResultE``,
        ``ResultE``).
    :param tasks: Variable number of callable tasks executed sequentially. Each
        task must accept the output of the previous task and return a Container or
        a compatible type.
    :param error_message: Custom error message included in the ``HTTPException`` if
        the pipeline fails at any step.
    :type error_message: str

    :returns: The unwrapped success value of the final pipeline result.
    :rtype: Any

    :raises HTTPException: Raised with status code 500 (INTERNAL_SERVER_ERROR) if
        any task in the pipeline fails or returns a failure Container. The exception
        detail contains the provided ``error_message``.

    :examples
    --------
    >>> def validate_user(data: dict) -> ResultE[dict]:
    ...     return Success(data) if data.get("id") else Failure("Invalid user")
    >>>
    >>> def enrich_user(data: dict) -> ResultE[dict]:
    ...     return Success({**data, "enriched": True})
    >>>
    >>> result = run_pipeline(
    ...     {"id": 123},
    ...     validate_user,
    ...     enrich_user,
    ...     error_message="User processing failed"
    ... )
    >>> # Returns: {"id": 123, "enriched": True}
    """
    return _capture_ioresult_value(
        _pipeline_flow(entry_value, *tasks),
        error_message,
    )
