"""
Mutator registry system for ScanImage transformations.

This module provides a registry-based pattern for managing image transformation
functions (mutators). All mutators used in pipelines must be registered,
ensuring traceability and validation.

Quick Start:
    from mutators import get_mutation_registry
    from returns.pipeline import flow

    registry = get_mutation_registry()

    @registry.register
    def my_filter(scan_image: ScanImage, param: float = 1.0) -> ScanImage:
        '''Apply custom filter to scan image.'''
        return scan_image.model_copy(update={"data": filtered_data})

    # Use in a pipeline with flow
    result = flow(
        scan_image,
        my_filter.bind(param=2.0),
        another_registered_mutator,
    )
"""

from mutators.composition import compose_mutators, create_conditional_mutator
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
    # Composition
    "compose_mutators",
    "create_conditional_mutator",
    # Exceptions
    "MutatorAlreadyRegisteredError",
]
