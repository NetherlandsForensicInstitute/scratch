"""
Immutable data container models for railway-oriented programming pipelines.

This module provides Pydantic-based data models that are propagated through railway
functions in functional pipelines. These models serve as type-safe, validated containers
for scientific and imaging data, ensuring data integrity as it flows through processing
pipelines.

Notes
-----
These models are designed specifically for railway-oriented programming where data
flows through a sequence of transformations. The immutability ensures that each
function in the pipeline receives unmodified input, preventing side effects and
making pipelines easier to reason about and debug.
"""

from .image import ImageContainer


__all__ = ["ImageContainer"]
