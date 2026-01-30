"""
Mutator registry system for ScanImage transformations.

This module provides a registry-based pattern for managing railway-oriented
image transformation functions (mutators). All mutators return ResultE[ScanImage].

Quick Start:
    from mutators import get_mutation_registry
    from returns.result import safe

    registry = get_mutation_registry()

    @registry.register
    @safe
    def my_filter(scan_image: ScanImage, param: float = 1.0) -> ScanImage:
        '''Apply custom filter to scan image.'''
        return scan_image.model_copy(update={"data": filtered_data})

    # Apply mutator
    result = my_filter(scan_image, param=2.0)
"""

from mutators.registry import get_mutation_registry
from mutators.types import (
    MutatorAlreadyRegisteredError,
    MutatorProtocol,
)

__all__ = [
    "get_mutation_registry",
    # Types
    "MutatorProtocol",
    # Exceptions
    "MutatorAlreadyRegisteredError",
]
