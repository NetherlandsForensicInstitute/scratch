import datetime as dt
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from x3p import X3Pfile

Array2D = np.ndarray[tuple[int, int], np.float32 | np.float64]


class ParsedImage(BaseModel):
    """Class for storing parsed scan data."""

    data: Array2D
    scale_x: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    scale_y: float = Field(default=1.0, gt=0.0, description="pixel size in um")
    filepath: os.PathLike | Path | None = None
    metadata: dict | None = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width_um(self) -> float:
        return self.scale_x * self.width

    @property
    def height_um(self) -> float:
        return self.scale_y * self.height

    def plot(self):
        """Generate a plot of the parsed image."""
        # TODO
        raise NotImplementedError

    def resample(self):
        """Resample the image data."""
        # TODO
        raise NotImplementedError

    def fill_na(self, mode: str = "linear", value: np.float64 | np.float32 | None = None):
        """
        Fill NaN-values in the image either by using interpolation or by replacing them with a fixed value.

        By default, the values will be filled using linear interpolation.

        :param mode: Interpolation mode, one of "linear" or "nearest"
        :param value: (Optional) The value used for filling. If set, no interpolation will be done.
        """
        # TODO
        raise NotImplementedError

    def to_x3p(self, filepath: os.PathLike | Path, author: str | None = None, comment: str | None = None):
        """Save the image data as an X3P file."""
        filepath = str(filepath)
        if os.path.exists(filepath):
            raise RuntimeError(f"Path already exists: {filepath}")

        output = X3Pfile()
        output.record1.set_featuretype("SUR")
        output.record1.axes.CX.set_axistype("I")
        output.record1.axes.CX.set_increment(self.scale_x)  # TODO: test if this should be in m or um
        output.record1.axes.CY.set_axistype("I")
        output.record1.axes.CY.set_increment(self.scale_y)  # TODO: test if this should be in m or um
        output.record2.set_date(dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%S"))
        if author:
            output.record2.set_creator(author)
        if comment:
            output.record2.set_comment(comment)
        # TODO: include more metadata in function arguments? (e.g. calibration date)
        output.record2.instrument.set_model("")
        output.record2.instrument.set_manufacturer("NFI")
        output.set_data(self.data)
        output.write(filepath)
