"""Global mutator registry and registration decorator."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from returns.result import ResultE

from mutators.types import (
    MutatorAlreadyRegisteredError,
    MutatorProtocol,
)

if TYPE_CHECKING:
    from container_models.scan_image import ScanImage


class _MutatorRegistry:
    """Global singleton registry for image mutators.

    This registry maintains a mapping of mutator names to their implementations
    and metadata. It enforces that only registered mutators can be used in
    validated pipelines.

    Example:
        >>> from returns.result import safe
        >>> registry = get_mutation_registry()
        >>> @registry.register
        ... @safe
        ... def gaussian_blur(scan_image: ScanImage, sigma: float = 1.0) -> ScanImage:
        ...     '''Apply Gaussian blur to scan image.'''
        ...     return scan_image.model_copy(update={"data": blurred_data})
    """

    _instance: _MutatorRegistry | None = None
    _mutators: dict[str, MutatorProtocol]

    def __new__(cls) -> _MutatorRegistry:
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._mutators = {}
        return cls._instance

    def register(self, function: MutatorProtocol) -> MutatorProtocol:
        """Register a mutator function with the global registry.

        Used as a decorator:

            @registry.register
            @safe
            def my_mutator(scan_image: ScanImage) -> ScanImage: ...

        :param function: The mutator function to register
        :returns: Wrapped mutator function
        :raises MutatorAlreadyRegisteredError: If name already registered
        """

        if function.__name__ in self._mutators:
            raise MutatorAlreadyRegisteredError(function.__name__)

        self._mutators[function.__name__] = function

        @wraps(function)
        def wrapper(scan_image: ScanImage, **kwargs: Any) -> ResultE[ScanImage]:
            return function(scan_image, **kwargs)

        return wrapper

    def __contains__(self, mutator: MutatorProtocol) -> bool:
        return mutator.__name__ in self._mutators

    def __len__(self) -> int:
        return len(self._mutators)


def get_mutation_registry() -> _MutatorRegistry:
    """Get the global mutator registry instance."""
    return _MutatorRegistry()
