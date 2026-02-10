"""Image mutations architecture.

This module defines how image modifications are structured and applied.

- :class:`~container_models.image.ImageContainer` holds depth data with scale metadata.
- :class:`ImageMutation` is an abstract interface for modifying an ImageContainer.
- Concrete mutations live in the ``mutations`` folder, each in its own file.

Architecture
------------
::

                    +------------------------------------+
                    |           ImageContainer           |
                    |------------------------------------|
                    | data     : DepthData               |
                    | metadata : MetaData                |
                    +------------------+-----------------+
                                       |
                                       v
                   +-------------------+----------------------+
                   |              <<abstract>>                |
                   |              ImageMutation               |
                   |------------------------------------------|
                   | + apply_on_image(T) -> T                 |
                   | + skip_predicate: bool                   |
                   +-------------------+----------------------+

Example
-------
.. code-block:: python

    from returns.pipeline import pipe
    from mutations.spatial import Resample, CropToMask
    from mutations.filter import LevelMap
    from container_models.base import Pair

    edit_pipeline = pipe(
        Resample(factors=Pair(2.0, 2.0)),
        CropToMask(mask=binary_mask),
        LevelMap(reference=Pair(0.0, 0.0), terms=SurfaceTerms.PLANE),
    )

    result = edit_pipeline(process_image)
"""

from abc import ABC, abstractmethod
from container_models import ImageContainer
from returns.result import safe


class ImageMutation[T: ImageContainer](ABC):
    """
    Represents a single mutation applied to an :class:`~container_models.image.ImageContainer`.

    After one `ImageMutation`, the resulting `ImageContainer` must be valid
    input for another mutation. This enables safe chaining in pipelines.

    Validation or skipping logic (for example: skipping resampling when
    scale factors are `(1, 1)`) can be implemented via `skip_predicate`.

    All parameters required for the mutation should be provided via
    the constructor.
    """

    @property
    def skip_predicate(self) -> bool:
        """
        Determines whether this mutation should be skipped.

        This can be used for simple validations. For example:
        - `Resample` may skip execution if both scale factors are `1`.

        :return bool:
            - `True`  → skip `apply_on_image`
            - `False` → apply the mutation
        """
        return False

    @safe
    def __call__(self, image: T) -> T:
        """
        Callable interface used by pipelines (e.g. `pipe(...)` from
        the `returns` library).

        If `skip_predicate` is `True`, the input `ImageContainer` is returned
        unchanged. Otherwise, `apply_on_image` is executed.

        :param image:
            The `ImageContainer` to be modified.
        :return ImageContainer:
            The resulting `ImageContainer`.
        """
        if self.skip_predicate:
            return image
        return self.apply_on_image(image)

    @abstractmethod
    def apply_on_image(self, image: T) -> T:
        """
        Applies the mutation to the given `ImageContainer`.

        This method must be implemented by concrete mutations and is
        called internally by `__call__` to support pipeline composition.

        :param image:
            The input `ImageContainer` to be modified.
        :return ImageContainer:
            A new or modified `ImageContainer`.
        """
