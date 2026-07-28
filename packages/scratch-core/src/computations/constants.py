from enum import IntEnum


class SurfaceTerms(IntEnum):
    """
    Surface fitting options exposed by the API.

    These options represent the surface leveling options and the values represent their respective polynomial degree.
    """

    NONE = 0
    PLANE = 1
    SPHERE = 2
