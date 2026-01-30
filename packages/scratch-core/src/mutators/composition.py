"""Utilities for composing mutators into reusable units."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from returns.pipeline import flow

from mutators.registry import get_mutation_registry
from mutators.types import MutatorProtocol, RegisteredMutator

if TYPE_CHECKING:
    from container_models.scan_image import ScanImage


def compose_mutators(
    *mutators: MutatorProtocol,
    name: str,
    description: str,
) -> RegisteredMutator:
    """Compose multiple mutators into a single registered mutator.

    This creates a new registered mutator that applies all provided mutators
    in sequence. The composed mutator is validated at creation time.

    Example:
        preprocessing = compose_mutators(
            level_surface.bind(terms=SurfaceTerms.PLANE),
            gaussian_blur.bind(sigma=1.0),
            normalize_intensity,
            name="standard_preprocessing",
            description="Level, blur, and normalize scan image.",
        )

        # Use like any other mutator
        result = preprocessing(scan_image)

    :param mutators: Mutators to compose (in execution order)
    :param name: Name for the composed mutator
    :param description: Description of the composed mutator
    :returns: A new registered mutator
    """

    # Register the composed mutator
    return get_mutation_registry().register(
        lambda scan_image: flow(scan_image, *mutators),
        name=name,
        description=description,
    )


def create_conditional_mutator(
    condition: Callable[[ScanImage], bool],
    if_true: MutatorProtocol,
    if_false: MutatorProtocol | None = None,
    *,
    name: str,
    description: str,
) -> RegisteredMutator:
    """Create a mutator that conditionally applies one of two mutators.

    Example:
        smart_filter = create_conditional_mutator(
            condition=lambda img: img.data.size > 1_000_000,
            if_true=fast_gaussian.bind(sigma=1.0),
            if_false=precise_gaussian.bind(sigma=1.0),
            name="smart_gaussian",
            description="Use fast filter for large images, precise for small.",
        )

    :param condition: Function that returns True/False based on ScanImage
    :param if_true: Mutator to apply when condition is True
    :param if_false: Mutator to apply when condition is False (identity if None)
    :param name: Name for the conditional mutator
    :param description: Description of the conditional mutator
    :returns: A new registered mutator
    """

    def conditional(scan_image: ScanImage) -> ScanImage:
        if condition(scan_image):
            return if_true(scan_image)
        if if_false:
            return if_false(scan_image)
        return scan_image

    return get_mutation_registry().register(
        conditional,
        name=name,
        description=description,
    )
