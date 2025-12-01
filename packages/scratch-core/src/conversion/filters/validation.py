import numpy as np

from conversion.filters.data_formats import FilterDomain


def _validate_regression_order(order: int) -> None:
    """Validate regression order parameter.

    :param order: Regression order to validate.
    :raises ValueError: If order is not 0, 1, or 2.
    """
    if order not in [0, 1, 2]:
        raise ValueError(f"regression_order must be 0, 1, or 2, got {order}")


def _validate_domain(domain: FilterDomain, allowed: tuple) -> None:
    """Validate domain parameter.

    :param domain: Domain to validate.
    :param allowed: Tuple of allowed FilterDomain values.
    :raises ValueError: If domain is not a FilterDomain enum or not in allowed values.
    """
    if not isinstance(domain, FilterDomain):
        raise ValueError(
            f"domain must be FilterDomain enum (FilterDomain.DISK or FilterDomain.RECTANGLE), "
            f"got {type(domain).__name__}: {domain}"
        )
    if domain not in allowed:
        allowed_names = ", ".join([f"FilterDomain.{d.name}" for d in allowed])
        raise ValueError(
            f"domain must be one of [{allowed_names}], got FilterDomain.{domain.name}"
        )


def _validate_filter_size(filter_size) -> None:
    """Validate filter size parameter.

    :param filter_size: Filter size (rows, cols).
    :raises ValueError: If filter_size is invalid.
    """
    size = np.asarray(filter_size, dtype=int).reshape(2)
    if np.any((size < 1) | (size % 2 == 0)):
        raise ValueError(f"filter_size must be odd and >= 1, got {size}")
