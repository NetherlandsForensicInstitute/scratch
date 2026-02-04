import numpy as np
from returns.io import IOResultE, IOSuccess
from returns.result import ResultE, Success

from container_models.base import UnitVector


def unwrap_result[T](result: IOResultE[T] | ResultE[T]) -> T:
    match result:
        case IOSuccess(Success(value)) | Success(value):
            return value
        case _:
            assert False, "failed to unwrap"


def spherical_to_unit_vector(azimuth: float, elevation: float) -> UnitVector:
    """Convert spherical coordinates (degrees) to a unit vector."""
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    return np.array(
        [
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad),
        ],
        dtype=np.float64,
    )
