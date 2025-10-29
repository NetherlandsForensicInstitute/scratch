import numpy as np
from surfalize.exceptions import CorruptedFileError
from surfalize.file.al3d import MAGIC, read_array, read_tag
from surfalize.file.common import RawSurface

"""
This file contains a modified copy of a method from the `surfalize` package. The original method uses tagged
values stored in the .al3d-file for obtaining the image size before parsing the binary image data. However,
some .al3d-files have an incorrect value stored for the image width, which leads to an incorrectly parsed image.
To fix this, we use the shape of the parsed buffer to compute the image width instead, before reshaping the data.

The original method can be found here:
https://github.com/fredericjs/surfalize/blob/d47b9b68636aae76e77329ac58ee0390765d7fb5/surfalize/file/al3d.py#L48
"""


def read_al3d(filehandle, read_image_layers=False, encoding="utf-8"):
    magic = filehandle.read(17)
    if magic != MAGIC:
        raise CorruptedFileError("Incompatible file magic detected.")
    header = dict()
    key, value = read_tag(filehandle, encoding=encoding)
    if key != "Version":
        raise CorruptedFileError("Version tag expected but not found.")
    header[key] = value

    key, value = read_tag(filehandle, encoding=encoding)
    if key != "TagCount":
        raise CorruptedFileError("TagCount tag expected but not found.")
    header[key] = value

    for _ in range(int(header["TagCount"])):
        key, value = read_tag(filehandle, encoding=encoding)
        header[key] = value

    # nx = int(header['Cols']) # ignore the 'Cols' tag since the value is often incorrect
    ny = int(header["Rows"])
    step_x = float(header["PixelSizeXMeter"]) * 1e6
    step_y = float(header["PixelSizeYMeter"]) * 1e6
    offset = int(header["DepthImageOffset"])
    filehandle.seek(offset)

    # original code
    # data = read_array(filehandle, dtype=np.float32, count=nx * ny, offset=0).reshape(ny, nx)

    # patched code
    offset_texture = int(header["TextureImageOffset"])
    # if no texture data is present, read until end of file or buffer
    count = offset_texture - offset if offset_texture > 0 else -1  # TODO: check if this is correct?
    data = read_array(filehandle, dtype=np.float32, count=count, offset=0)
    # compute `nx` from the data shape
    nx = data.shape[0] // ny
    data = data.reshape(ny, nx)

    invalidValue = float(header["InvalidPixelValue"])
    data[data == invalidValue] = np.nan

    data *= 1e6  # Conversion from m to um

    return RawSurface(data, step_x, step_y)
