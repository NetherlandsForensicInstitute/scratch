"""Type definitions for the mutator registry system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from returns.result import ResultE

if TYPE_CHECKING:
    from container_models.scan_image import ScanImage


@runtime_checkable
class MutatorProtocol(Protocol):
    """Protocol defining the expected signature for image mutators.

    Railway-oriented mutators accept a ScanImage and return Result[ScanImage, Exception].
    Use @safe decorator to wrap functions that may raise exceptions.
    """

    def __call__(self, scan_image: ScanImage, **kwargs: Any) -> ResultE[ScanImage]: ...
    @property
    def __name__(self) -> str: ...


class MutatorAlreadyRegisteredError(Exception):
    """Raised when attempting to register a mutator with a name that already exists."""

    def __init__(self, mutator_name: str) -> None:
        self.mutator_name = mutator_name
        super().__init__(
            f"Mutator '{mutator_name}' is already registered. "
            f"Use a different name or unregister the existing one first."
        )
