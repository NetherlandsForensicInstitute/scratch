"""
Image Modifications Architecture
================================

This module defines how image modifications are structured and applied.

- `ScanImage` represents the scanned image as a complete data package
  (image data + metadata).
- `ImageMutation` is an abstract interface for modifying a `ScanImage`.
- Concrete mutations (e.g. Resample, Crop, Mask, Scale) live in the
  `mutations` folder, each in its own file.
- Loosely coupled or stateless functionality (such as solvers or pure
  computations) should live in the `computations` folder.

High-level Design
-----------------

                 +------------------------------------------------+
                 |                  ScanImage                     |
                 |------------------------------------------------|
                 | data     : np.ndarray                          |
                 | scale_x  : float                               |
                 | scale_y  : float                               |
                 |------------------------------------------------|
                 | modification : Modification                    |
                 +-----------------------+------------------------+
                                         |
                                         v
                     +-------------------+---------------------+
                     |               <<abstract>>              |
                     |               Modification              |
                     |-----------------------------------------|
                     | + apply_on_image(ScanImage) -> ScanImage|
                     +-------------------+---------------------+
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

    from returns.pipeline import pipe
    from numpy import ones, float64
    from mutations import (
        Resample,
        Mask,
        LevelMap,
        GaussianFilter,
    )

    scan_image = ScanImage(
        data=ones((10, 10), float64),
        scale_x=1,
        scale_y=1,
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

    result = edit_image_pipeline(scan_image)
"""

from abc import ABC, abstractmethod
from container_models.scan_image import ScanImage
from returns.result import safe


class ImageMutation(ABC):
    """
    Represents a single mutation applied to a `ScanImage`.

    After one `ImageMutation`, the resulting `ScanImage` must be valid
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
    def __call__(self, scan_image: ScanImage) -> ScanImage:
        """
        Callable interface used by pipelines (e.g. `pipe(...)` from
        the `returns` library).

        If `skip_predicate` is `True`, the input `ScanImage` is returned
        unchanged. Otherwise, `apply_on_image` is executed.

        :param scan_image:
            The `ScanImage` to be modified.
        :return ScanImage:
            The resulting `ScanImage`.
        """
        if self.skip_predicate:
            return scan_image
        return self.apply_on_image(scan_image=scan_image)

    @abstractmethod
    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Applies the mutation to the given `ScanImage`.

        This method must be implemented by concrete mutations and is
        called internally by `__call__` to support pipeline composition.

        :param scan_image:
            The input `ScanImage` to be modified.
        :return ScanImage:
            A new or modified `ScanImage`.
        """
