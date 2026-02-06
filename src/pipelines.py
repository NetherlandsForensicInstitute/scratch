"""Pipeline execution utilities.

This module provides utilities for executing image mutation pipelines.

Example
-------
.. code-block:: python

    from returns.pipeline import pipe
    from mutations.spatial import Resample
    from mutations.shading import LightIntensityMap

    result = run_pipeline(
        pipe(Resample(factors), LightIntensityMap(lights, observer)),
        image,
        "Processing failed",
    )
"""

from collections.abc import Callable
from http import HTTPStatus

from container_models.image import ImageContainer
from fastapi import HTTPException
from returns.result import ResultE, Success


def run_pipeline[T: ImageContainer](pipe: Callable[[T], ResultE[T]], image: T, error_message: str) -> T:
    """Execute a sequence of mutations on a given image.

    :param pipe: A composed pipeline of mutations.
    :param image: The input image to process.
    :param error_message: Message for HTTPException on failure.
    :returns: The processed image.
    :raises HTTPException: If the pipeline fails.
    """
    match pipe(image):
        case Success(value):
            return value
        case _:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=error_message)
