"""
Mutator registry system for ScanImage transformations.

This module provides a registry-based pattern for managing railway-oriented
image transformation functions (mutators). All mutators return ResultE[ScanImage]
and chain using .bind() for railway-oriented programming.

Quick Start:
    from mutators import get_mutation_registry
    from returns.curry import partial
    from returns.result import Success, safe

    registry = get_mutation_registry()

    @registry.register
    @safe
    def my_filter(scan_image: ScanImage, param: float = 1.0) -> ScanImage:
        '''Apply custom filter to scan image.'''
        return scan_image.model_copy(update={"data": filtered_data})

    # Chain mutators with .bind()
    result = (
        Success(scan_image)
        .bind(partial(my_filter, param=2.0))
        .bind(another_registered_mutator)
    )
"""

from mutators.registry import get_mutation_registry
from mutators.types import (
    MutatorAlreadyRegisteredError,
    MutatorMetadata,
    MutatorProtocol,
)

__all__ = [
    "get_mutation_registry",
    # Types
    "MutatorMetadata",
    "MutatorProtocol",
    # Exceptions
    "MutatorAlreadyRegisteredError",
]
