"""
Image Modifications Architecture
================================

This module defines how image modifications are structured and applied.

- :class:`~container_models.image.ImageContainer` holds depth data with scale metadata.
- :class:`ImageMutation` is an abstract interface for modifying an ImageContainer.
- Concrete mutations live in the ``mutations`` folder, each in its own file.
- Loosely coupled or stateless functionality (such as solvers or pure
  computations) should live in the `computations` folder.

High-level Design
-----------------

                        +---------------------------------+
                        |           ImageContainer        |
                        |---------------------------------|
                        | data     : DepthData            |
                        | metadata : MetaData             |
                        +---------------+-----------------+
                                        |
                                        v
                    +-------------------+----------------------+
                    |              <<abstract>>                |
                    |              ImageMutation               |
                    |------------------------------------------|
                    | + apply_on_image(T) -> T                 |
                    | + skip_predicate: bool                   |
                    +--------------------+---------------------+
                                         ^
                                         |
        +---------------------+----------+------------+-------------------+
        |                     |                       |                   |
        |                     |                       |                   |
+-------+--------+  +---------+---------+  +----------+--------+  +-------+-------+
|   Resample     |  |       Crop        |  |         Mask      |  |     Scale     |
|----------------|  |-------------------|  |-------------------|  |---------------|
| x : int        |  | area : np.ndarray |  | area : np.ndarray |  | x : int       |
| y : int        |  |                   |  |                   |  | y : int       |
+----------------+  +-------------------+  +-------------------+  +---------------+
| <<overwrite>>  |  |   <<overwrite>>   |  |   <<overwrite>>   |  | <<overwrite>> |
| apply_on_image |  |  apply_one_image  |  |   apply_on_image  |  | apply_on_image|
+----------------+  |  skip_predicate   |  +-------------------+  +---------------+
                    +-------------------+


Example
-------

    from container_models.image import ImageContainer, MetaData
    from container_models.base import Pair
    from returns.pipeline import pipe
    from numpy import ones, float64
    from mutations import (
        Resample,
        Mask,
        LevelMap,
        GaussianFilter,
    )

    image = ScanImage(
        data=ones((10, 10), float64),
        metadata=MetaData(scale=Pair(1, 1))
    )

    edit_image_pipeline = pipe(
        Resample(factors=Point(2, 2)),
        Mask(mask=np.zeros((5, 5), dtype=bool)),
        LevelMap(
            terms=SurfaceTerms.ASTIG_0,
            solver=solve_least_squares,
            reference_point=Point(3, 3),
        ),
        GaussianFilter(
            cutoff_pixels=np.ones((5, 5)),
            regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        ),
    )

    result = edit_image_pipeline(image)
"""

from abc import ABC, abstractmethod

from returns.result import safe

from container_models import ImageContainer


class ImageMutation(ABC):
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
    def __call__(self, image: ImageContainer) -> ImageContainer:
        """
        Callable interface used by pipelines (e.g. `pipe(...)` from
        the `returns` library).

        If `skip_predicate` is `True`, the input `ImageContainer` is returned
        unchanged. Otherwise, `apply_on_image` is executed.

        :param image:
            The `ImageContainer` to be modified.
        :return ImageContainer:
            The same image object passed as input.
        """
        if self.skip_predicate:
            return image
        return self.apply_on_image(image)

    @abstractmethod
    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Applies the mutation to the given `ImageContainer`.

        This method must be implemented by concrete mutations and is
        called internally by `__call__` to support pipeline composition.

        :param image:
            The input `ImageContainer` to be modified.
        :return ImageContainer:
            A new or modified `ImageContainer`.
        """
