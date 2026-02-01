"""Type definitions for the mutator registry system."""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Protocol

from loguru import logger
from pydantic import BaseModel
from returns.pipeline import is_successful
from returns.result import ResultE, Success

from image_tasks.types.scan_image import ScanImage


class ImageTask(Protocol):
    def __call__(
        self, scan_image: ScanImage, *args, **kwargs
    ) -> ResultE[ScanImage]: ...
    @property
    def __name__(self) -> str: ...


class ImageTaskContext[P: BaseModel](BaseModel):
    name: str
    skip_predicate: Callable[..., bool] | None = None
    skip_on_error: bool = False
    params: P | None = None

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params) if self.params else {}


class AbstractImageTask[C: ImageTaskContext](ABC):
    def __init__(self, task: ImageTask, context: C) -> None:
        self.task = task
        self.context = context

    def _update_scan_image_history(self, scan_image: ScanImage) -> ScanImage:
        scan_image.meta_data.processing_history.append(self.context.name)
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
