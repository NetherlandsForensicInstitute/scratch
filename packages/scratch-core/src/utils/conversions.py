import numpy as np


def convert_azimuth_elevation_to_vector(az, el):
    """Compute vector from azimuth/elevation (degrees)."""
    azr = np.deg2rad(az)
    elr = np.deg2rad(el)
    v = np.array([-np.cos(azr) * np.cos(elr), np.sin(azr) * np.cos(elr), np.sin(elr)])
    return v
