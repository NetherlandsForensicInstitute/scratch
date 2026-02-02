from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Any, Protocol

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


type Predicate = Callable[..., bool]


class ImageTaskContext[P: BaseModel](BaseModel):
    name: str
    skip_predicate: Predicate | None
    alternative_predicate: Predicate | None
    skip_on_error: bool
    params: P | None

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params) if self.params else {}


class AbstractImageTask[C: ImageTaskContext](ABC):
    def __init__(
        self, task: ImageTask, context: C, alternative_task: ImageTask | None
    ) -> None:
        self.task = task
        self.context = context
        if alternative_task and not context.alternative_predicate:
            raise ValueError("Alternative Predicate is missing")
        self.alternative_task = alternative_task

    def _update_scan_image_history(self, scan_image: ScanImage) -> ScanImage:
        scan_image.meta_data.processing_history.append(self.context.name)
        return scan_image

    def __call__(self, scan_image: ScanImage) -> ResultE[ScanImage]:
        if self.context.skip_predicate and self.context.skip_predicate(
            scan_image=scan_image, **self.context.kwargs
        ):
            logger.info(f"No {self.context.name} needed")
            return Success(scan_image)
        if (
            self.alternative_task
            and self.context.alternative_predicate
            and self.context.alternative_predicate(scan_image, **self.context.kwargs)
        ):
            return self.alternative_task(scan_image, **self.context.kwargs)
        if is_successful(result := self.task(scan_image, **self.context.kwargs)):
            return result.map(self._update_scan_image_history)
        if self.context.skip_on_error:
            return result.from_value(scan_image).map(self._update_scan_image_history)
        return result
