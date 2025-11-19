import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray
from skimage.transform import resize
from surface_conversion import convert_image_to_slope_map
from surface_conversion.translations import calculate_surface
from utils.conversions import convert_azimuth_elevation_to_vector


def get_surface_plot(data_in, *args):
    """
    Python translation of MATLAB function GetSurfacePlot.
    ----------------------------------------------------
    See MATLAB header for full description.
    """
    # User settings
    doplot = 0  # 0/1 = no/yes (only for development purposes)
    light_angles = np.array([[90, 45], [180, 45]])  # default light sources [az el]

    # Retrieve data needed, forget the rest
    depthdata = data_in["depth_data"]
    xdim = data_in["xdim"]
    ydim = data_in["ydim"]

    # Plot depthdata (placeholder, matplotlib if doplot)
    if doplot:
        plt.figure()
        plt.imshow(depthdata, cmap="gray")
        plt.axis("equal")
        plt.show()

    # Turn an extra light source on if requested
    if len(args) > 0 and args[0] is not None:
        light_angles = np.array([[90, 45], [180, 45], [270, 45]])

    # If light sources are provided, these replace the defaults
    if len(args) > 1 and args[1] is not None and len(args[1]) > 0:
        light_angles = np.array(args[1])

    # If a mask is provided, mask the image before generating the surface
    if len(args) > 2 and args[2] is not None and np.any(args[2]):
        mask = args[2]

        # Resize mask if dimensions mismatch
        if mask.shape != depthdata.shape:
            # Equivalent to imresize(mask, size(depthdata), 'nearest')
            mask = resize(
                mask, depthdata.shape, order=0, preserve_range=True, anti_aliasing=False
            )

        # Mask background data with NaN
        depthdata = np.where(mask, depthdata, np.nan)
    else:
        mask = np.ones_like(depthdata)  # everything is foreground

    # Set fill_display
    if len(args) > 3:
        fill_display = args[3]
    else:
        fill_display = 0

    # If the fill display option is set, remove the boundaries
    if fill_display:
        bbox = DetermineBoundingBox(mask)
        depthdata = depthdata[bbox[1, 0] - 1 : bbox[1, 1], bbox[0, 0] - 1 : bbox[0, 1]]

    n1, n2, n3 = convert_image_to_slope_map(depthdata=depthdata, xdim=xdim, ydim=ydim)

    # Calculate intensity of surface for each light source
    Iout = merge_depth_map_with_slope_maps(depthdata, n1, n2, n3, light_angles)

    # Calculate total intensity of surface
    Iout = np.nansum(Iout, axis=2)

    # Normalize between [0,1]
    Imin = np.nanmin(Iout)
    Imax = np.nanmax(Iout)
    Iout = (Iout - Imin) / (Imax - Imin)

    # Add ambient component and scale [0,1]->[0,255]
    famb = 25
    Iout = famb + (255 - famb) * Iout

    return Iout


def merge_depth_map_with_slope_maps(
    depthdata: NDArray[tuple[int, int]],
    n1: NDArray[tuple[int, int]],
    n2: NDArray[tuple[int, int]],
    n3: NDArray[tuple[int, int]],
    light_angles,
):
    """

    Parameters
    ----------
    depthdata
    n1
    n2
    n3
    light_angles

    Returns
    -------

    """
    nLS = light_angles.shape[0]
    Iout = np.full(
        (*depthdata.shape, nLS), np.nan
    )  # adds sort of empty layer per light_angle to fill later on
    # Create vector pointing towards observer
    OBS = convert_azimuth_elevation_to_vector(0, 90)  # azimuth 0 deg, elevation 90 deg
    for i in range(nLS):
        LS = convert_azimuth_elevation_to_vector(
            light_angles[i, 0], light_angles[i, 1]
        )  # get light el and az
        Iout[:, :, i] = calculate_surface(LS, OBS, n1, n2, n3)
    return Iout


def DetermineBoundingBox(mask):
    """Determine bounding box of mask."""
    x_sum = np.sum(mask, axis=0)
    y_sum = np.sum(mask, axis=1)

    start_x = np.argmax(x_sum > 0) + 1
    end_x = len(x_sum) - np.argmax(x_sum[::-1] > 0)
    start_y = np.argmax(y_sum > 0) + 1
    end_y = len(y_sum) - np.argmax(y_sum[::-1] > 0)

    bounding_box = np.array([[start_x, end_x], [start_y, end_y]])
    return bounding_box
