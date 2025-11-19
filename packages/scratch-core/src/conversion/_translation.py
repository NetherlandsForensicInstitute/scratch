import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from surface_conversion.translations import pre_refactor_logic
from surface_conversion.schemas import LightAngle


def get_surface_plot(data_in, *args):
    """
    Python translation of MATLAB function GetSurfacePlot.
    ----------------------------------------------------
    See MATLAB header for full description.
    """
    # User settings
    doplot = 0  # 0/1 = no/yes (only for development purposes)
    light_angles = (
        LightAngle(azimuth=90, elevation=45),
        LightAngle(azimuth=180, elevation=45),
    )
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

    # Calculate intensity of surface for each light source
    Iout = pre_refactor_logic(depthdata, xdim, ydim, light_angles)

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
