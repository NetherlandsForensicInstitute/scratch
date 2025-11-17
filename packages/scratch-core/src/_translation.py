import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray
from skimage.transform import resize
from scipy.signal import convolve2d

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


def merge_depth_map_with_slope_maps(depthdata, n1, n2, n3, light_angles):
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
    OBS = getv(0, 90)  # azimuth 0 deg, elevation 90 deg
    for i in range(nLS):
        LS = getv(light_angles[i, 0], light_angles[i, 1])  # get light el and az
        Iout[:, :, i] = calcsurf(LS, OBS, n1, n2, n3)
    return Iout


def convert_image_to_slope_map(
    depthdata: NDArray[tuple[int, int]], xdim: int, ydim: int
):
    """
    Compute surface-normal components (n1, n2, n3) from a 2D depth map.

    Parameters
    ----------
    depthdata : NDArray
        2-D array of depth values, indexed as depth[y, x].
    xdim : float
        Physical spacing between columns in meters (Δx).
    ydim : float
        Physical spacing between rows in meters (Δy).

    Returns
    -------
    n1, n2, n3 : NDArray
        Components of the unit surface normal.
        n1 = -∂z/∂x / norm
        n2 = -∂z/∂y / norm
        n3 = 1 / norm
    """
    factor_x = 1 / (2 * xdim)
    factor_y = 1 / (2 * ydim)
    K = [[1, 0, -1]]
    hx = convolve2d(depthdata, K, 'same', fillvalue=np.nan) * factor_x
    hy = convolve2d(depthdata.T, K, 'same', fillvalue=np.nan).T * factor_y

    # # Create surface normals
    # hx = np.diff(depthdata, axis=1) / xdim  # slope in x-direction (∂z/∂x)
    # hy = np.diff(depthdata, axis=0) / ydim  # slope in y-direction (∂z/∂y)
    #
    # # Extend to match original dimensions
    # hx = np.hstack([hx, np.full((hx.shape[0], 1), np.nan)])  # pad last column
    # hy = np.vstack([hy, np.full((1, hy.shape[1]), np.nan)])  # pad last row

    norm = np.sqrt(hx * hx + hy * hy + 1)
    n1 = -hx / norm
    n2 = hy / norm
    n3 = 1 / norm
    return n1, n2, n3


# --------------------------------------------
def getv(az, el):
    """Compute vector from azimuth/elevation (degrees)."""
    azr = np.deg2rad(az)
    elr = np.deg2rad(el)
    v = np.array(
        [-np.cos(azr) * np.cos(elr), np.sin(azr) * np.cos(elr), np.sin(elr)]
    )  # vx,vy,vz as 3D light vector.
    return v


# --------------------------------------------
def calcsurf(LS, OBS, n1, n2, n3):
    """Calculate diffuse + specular components."""
    # PREPARATIONS
    h = LS + OBS
    h = h / np.sqrt(np.dot(h, h))  # normalize

    # DIFFUSE COMPONENT
    Idiffuse = LS[0] * n1 + LS[1] * n2 + LS[2] * n3
    Idiffuse[Idiffuse < 0] = 0

    # SPECULAR COMPONENT
    Ispec = h[0] * n1 + h[1] * n2 + h[2] * n3
    Ispec[Ispec < 0] = 0
    Ispec = np.cos(2 * np.arccos(Ispec))
    Ispec[Ispec < 0] = 0

    # Phong factor f=4
    Ispec = Ispec**4

    # Combine
    fspec = 1
    Iout = (Idiffuse + fspec * Ispec) / (1 + fspec)
    return Iout


# --------------------------------------------
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
