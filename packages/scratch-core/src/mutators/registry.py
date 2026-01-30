"""Global mutator registry and registration decorator."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, overload

from mutators.types import (
    MutatorAlreadyRegisteredError,
    MutatorMetadata,
    MutatorProtocol,
    RegisteredMutator,
)

if TYPE_CHECKING:
    from container_models.scan_image import ScanImage

type WrapRegisterMutator = Callable[[MutatorProtocol], RegisteredMutator]


class _MutatorRegistry:
    """Global singleton registry for image mutators.

    This registry maintains a mapping of mutator names to their implementations
    and metadata. It enforces that only registered mutators can be used in
    validated pipelines.

    Example:
        >>> registry = get_mutation_registry()
        >>> @registry.register
        ... def gaussian_blur(scan_image: ScanImage, sigma: float = 1.0) -> ScanImage:
        ...     '''Apply Gaussian blur to scan image.'''
        ...     return scan_image.model_copy(update={"data": blurred_data})
    """

    _instance: _MutatorRegistry | None = None
    _mutators: dict[str, RegisteredMutator]

    def __new__(cls) -> _MutatorRegistry:
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._mutators = {}
        return cls._instance

    @overload
    def register(self, function: MutatorProtocol) -> RegisteredMutator: ...

    @overload
    def register(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> WrapRegisterMutator: ...

    def register(
        self,
        function: MutatorProtocol | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> RegisteredMutator | WrapRegisterMutator:
        """Register a mutator function with the global registry.

        Can be used as a decorator with or without arguments:

            @registry.register
            def my_mutator(scan_image: ScanImage) -> ScanImage: ...

            @registry.register(name="custom_name")
            def my_mutator(scan_image: ScanImage) -> ScanImage: ...

        :param function: The mutator function to register
        :param name: Override the mutator name (defaults to function name)
        :param description: Override description (defaults to docstring)
        :returns: RegisteredMutator wrapping the function
        :raises MutatorAlreadyRegisteredError: If name already registered
        """

        def decorator(func: MutatorProtocol) -> RegisteredMutator:
            mutator_name = name or func.__name__

            if mutator_name in self._mutators:
                raise MutatorAlreadyRegisteredError(mutator_name)

            metadata = MutatorMetadata(
                name=mutator_name,
                description=description or (func.__doc__ or "").strip().split("\n")[0],
            )

            @wraps(func)
            def wrapper(scan_image: ScanImage, **kwargs: Any) -> ScanImage:
                return func(scan_image, **kwargs)

            registered = RegisteredMutator(func=wrapper, metadata=metadata)
            self._mutators[mutator_name] = registered

            return registered

        if function is not None:
            return decorator(function)
        return decorator

    def list_mutators(self) -> list[MutatorMetadata]:
        """List all registered mutators.

        :returns: List of mutator metadata
        """
        return [mutator.metadata for mutator in self._mutators.values()]

    def __contains__(self, value: str | MutatorProtocol) -> bool:
        if callable(value):
            value = getattr(value, "name", value.__name__)
        return value in self._mutators

    def __len__(self) -> int:
        return len(self._mutators)


def get_mutation_registry() -> _MutatorRegistry:
    """Get the global mutator registry instance."""
    return _MutatorRegistry()
