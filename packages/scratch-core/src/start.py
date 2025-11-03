from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field
from skimage.transform import resize


class UserSettings(BaseModel):
    """User settings, doplot is only for development purposes."""

    doplot: bool = Field(False, description="0/1 = no/yes (only for development purposes)")
    light_angles: Literal[(90, 45), (180, 45)] = Field(..., description="azimuth, elevation of each light source [deg]")


class DataRetrieved(BaseModel):
    """some data that is probably we get from a different location already."""

    depthdata: np.array = Field(..., description="depth data from this location, no CLUE ABOUT THE DATA TYPE")
    xdim: int = Field(..., description="x dimension of data, no CLUE ABOUT THE DATA TYPE")
    ydim: int = Field(..., description="y dimension of data, no CLUE ABOUT THE DATA TYPE")


"""when DaataRetrieved is got in then the clear data_in is used"""


def one_big_function(user_settings: UserSettings, data_retrieved: DataRetrieved, varargin):
    if user_settings.doplot:
        plt.figure()
        plt.imshow(user_settings.depthdata, cmap="gray", aspect="equal")
        plt.axis("on")
        plt.show()

    if len(varargin) > 0 and varargin[0] is not None:
        np.array([[90, 45], [180, 45], [270, 45]])

    if len(varargin) > 1 and varargin[1] is not None:
        varargin[1]

    if len(varargin) > 2 and varargin[2] is not None:
        mask = varargin[2]

        if mask.shape != data_retrieved.depthdata.shape:
            mask = resize(mask, data_retrieved.depthdata.shape, preserve_range=True, anti_aliasing=True)
        data_retrieved.depthdata[-mask.astype(bool)] = None
    else:
        mask = np.ones(data_retrieved.depthdata.shape, dtype=bool)

    fill_display = 0
    if len(varargin) > 3:
        fill_display = varargin[3]

    if fill_display:
        bbox = determine_bounding_box(mask)
        data_retrieved.depthdata[bbox[1, 0] : bbox[1, 1], bbox[0, 0] : bbox[0, 1]]


def determine_bounding_box(mask): ...
