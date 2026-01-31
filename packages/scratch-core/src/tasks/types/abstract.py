"""Type definitions for the mutator registry system."""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel
from returns.pipeline import is_successful

from container_models.scan_image import ScanImage
from returns.result import ResultE, Success, safe


@runtime_checkable
class ImageTask(Protocol):
    """Protocol defining the expected signature for image mutators.

    Railway-oriented mutators accept a ScanImage and return Result[ScanImage, Exception].
    Use @safe decorator to wrap functions that may raise exceptions.
    """

    @safe
    def __call__(self, scan_image: ScanImage, **kwargs: Any) -> ScanImage: ...


class ImageTaskContext[P: BaseModel](BaseModel):
    name: str
    skip_predicate: Callable[..., bool] | None = None
    skip_on_error: bool = False
    params: P | None = None

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.params.model_dump(exclude={"predicate"}) if self.params else {}


class AbstractImageTask[C: ImageTaskContext, T: ImageTask](ABC):
    def __init__(self, task: T, context: C) -> None:
        self.task = task
        self.context = context

    def _update_scan_image_history(self, scan_image: ScanImage) -> ScanImage:
        scan_image.meta_data.processing_history.append(str(self.task))
        return scan_image

    def __call__(self, scan_image: ScanImage) -> ResultE[ScanImage]:
        if self.context.skip_predicate and self.context.skip_predicate(
            scan_image=scan_image, **self.context.kwargs
        ):
            logger.info(f"No {self.context.name} needed")
            return Success(scan_image)
        if is_successful(result := self.task(scan_image, **self.context.kwargs)):
            return result.map(self._update_scan_image_history)
        if self.context.skip_on_error:
            return result.from_value(scan_image).map(self._update_scan_image_history)
        return result
