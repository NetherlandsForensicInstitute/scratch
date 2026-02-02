from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel
from returns.result import ResultE

from image_tasks.types.abstract import (
    AbstractImageTask,
    ImageTask,
    ImageTaskContext,
    Predicate,
)
from image_tasks.types.scan_image import ScanImage
from utils.logger import log_railway_function


class ImageTaskFactory(Protocol):
    """Protocol for the factory class returned by create_image_task."""

    def __call__(self, **kwargs: Any) -> AbstractImageTask: ...


def create_image_task[P: BaseModel](
    task: ImageTask,
    alternative: ImageTask | None = None,
    *,
    params_model: type[P] | None = None,
    skip_predicate: Predicate | None = None,
    alternative_predicate: Predicate | None = None,
    skip_on_error: bool = False,
    failure_msg: str = "Task failed",
    success_msg: str = "Task succeeded",
) -> ImageTaskFactory:
    """
    Create an image task class from a task function.

    :param task: The task function to wrap. Must accept ScanImage and return ResultE[ScanImage].
    :param alternative: An alternative task function to execute when alternative_predicate is True.
    :param params_model: Pydantic model for task parameters. If None, task takes no params.
    :param skip_predicate: Function that determines if task should be skipped entirely.
    :param alternative_predicate: Function that determines if alternative task should run instead.
    :param skip_on_error: If True, continue pipeline on task failure.
    :param failure_msg: Log message on failure.
    :param success_msg: Log message on success.
    :returns: A factory that creates task instances when called with parameters.

    Example::

        @safe
        def _subsample_scan_image(
            scan_image: ScanImage, *, step_size_x: int, step_size_y: int
        ) -> ScanImage:
            '''Subsample the scan image by given step sizes.'''
            ...

        SubsampleScanImage = create_image_task(
            _subsample_scan_image,
            params_model=SubSampleParams,
            skip_predicate=lambda step_size_x, step_size_y, **_: (
                step_size_x == 1 and step_size_y == 1
            ),
            failure_msg="Failed to subsample image",
            success_msg="Successfully subsampled image",
        )

        task = SubsampleScanImage(step_size_x=2, step_size_y=2)
        result = task(some_scan_image)
    """
    task_name = task.__name__.strip("_")
    alternative_name = f"_or_{alternative.__name__.strip('_')}" if alternative else ""
    name = f"{task_name}{alternative_name}"

    class FactoryCreatedTask(AbstractImageTask):
        def __init__(self, **kwargs: Any) -> None:
            params = params_model(**kwargs) if params_model and kwargs else None
            context = ImageTaskContext(
                name=name,
                params=params,
                skip_predicate=skip_predicate,
                alternative_predicate=alternative_predicate,
                skip_on_error=skip_on_error,
            )
            super().__init__(task, context, alternative)

        @log_railway_function(failure_msg, success_msg)
        def __call__(self, scan_image: ScanImage) -> ResultE[ScanImage]:
            return super().__call__(scan_image)

    FactoryCreatedTask.__name__ = name.title().replace("_", "")
    FactoryCreatedTask.__qualname__ = FactoryCreatedTask.__name__
    FactoryCreatedTask.__doc__ = task.__doc__

    return FactoryCreatedTask
