from typing import Annotated
from enum import StrEnum
import json

import numpy as np
from lir import LLRData

from container_models.base import (
    FloatArray2D,
)

from functools import partial
from pydantic import (
    Field,
    computed_field,
    AfterValidator,
    PlainSerializer,
    BeforeValidator,
    model_validator,
)
from numpy import float64
from container_models.base import (
    ConfigBaseModel,
    coerce_to_array,
    serialize_ndarray,
)
from container_models.scan_image import ScanImage


class MarkType(StrEnum):
    # Impression marks
    BREECH_FACE_IMPRESSION = "breech face impression mark"
    CHAMBER_IMPRESSION = "chamber impression mark"
    EJECTOR_IMPRESSION = "ejector impression mark"
    EXTRACTOR_IMPRESSION = "extractor impression mark"
    FIRING_PIN_IMPRESSION = "firing pin impression mark"

    # Striation marks
    APERTURE_SHEAR_STRIATION = "aperture shear striation mark"
    BULLET_GEA_STRIATION = "bullet gea striation mark"
    BULLET_LEA_STRIATION = "bullet lea striation mark"
    CHAMBER_STRIATION = "chamber striation mark"
    EJECTOR_STRIATION = "ejector striation mark"
    EJECTOR_PORT_STRIATION = "ejector port striation mark"
    EXTRACTOR_STRIATION = "extractor striation mark"
    FIRING_PIN_DRAG_STRIATION = "firing pin drag striation mark"

    def is_impression(self) -> bool:
        return "IMPRESSION" in self.name

    def is_striation(self) -> bool:
        return "STRIATION" in self.name

    @property
    def scale(self) -> float:
        if self == MarkType.BREECH_FACE_IMPRESSION:
            return 3.5e-6
        return 1.5e-6


def validate_rectangle_corners(arr: FloatArray2D) -> FloatArray2D:
    """Validate that array has shape (4, 2)"""
    if arr.shape != (4, 2):
        raise ValueError(f"Rectangle must have shape (4, 2), got {arr.shape}")
    return arr


# Note: Our code expects pixel coordinates, i.e. top-left origin, in the order [x, y]
BoundingBox = Annotated[
    FloatArray2D,
    BeforeValidator(partial(coerce_to_array, float64)),
    AfterValidator(partial(validate_rectangle_corners)),
    PlainSerializer(serialize_ndarray),
]


class Mark(ConfigBaseModel):
    """
    Representation of a mark (impression or striation)
    """

    scan_image: ScanImage
    mark_type: MarkType
    meta_data: dict = Field(default_factory=dict)
    center_: tuple[float, float] | None = Field(default=None, alias="center")

    @computed_field
    @property
    def center(self) -> tuple[float, float]:
        """
        Center point of the mark in image coordinates.

        Returns the center as (x, y) where x is the horizontal position
        (column) and y is the vertical position (row). If no explicit
        center has been set, computes it as the geometric center of the
        scan image.

        :returns: Center coordinates as (x, y),
        """
        if self.center_ is not None:
            return self.center_
        data = self.scan_image.data
        return data.shape[1] / 2, data.shape[0] / 2

    def export(self) -> str:
        """Export the `Mark` meta-data fields as a JSON string."""
        data = {
            "mark_type": self.mark_type.name,
            "center": self.center,
            "scale_x": self.scan_image.scale_x,
            "scale_y": self.scan_image.scale_y,
            "meta_data": self.meta_data,
        }
        return json.dumps(data, indent=4)


class ReferenceData(ConfigBaseModel):
    km_model: str
    km_scores: np.ndarray
    km_llr_data: LLRData
    knm_model: str
    knm_scores: np.ndarray
    knm_llr_data: LLRData

    @model_validator(mode="after")
    def _validate_matching_lengths(self):
        if len(self.km_scores) != len(self.km_llr_data.llrs):
            raise ValueError("km_scores and km_lrs must have the same length")
        if len(self.knm_scores) != len(self.knm_llr_data.llrs):
            raise ValueError("knm_scores and knm_lrs must have the same length")
        return self

    @property
    def scores(self) -> np.ndarray:
        return np.concatenate([self.km_scores, self.knm_scores])

    @property
    def llrs(self) -> np.ndarray:
        return np.concatenate([self.km_llr_data.llrs, self.knm_llr_data.llrs])

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate(
            [
                np.ones(len(self.km_scores)),
                np.zeros(len(self.knm_scores)),
            ]
        )

    @property
    def llrs_at5(self) -> np.ndarray:
        km = self.km_llr_data.llr_intervals
        knm = self.knm_llr_data.llr_intervals
        return np.concatenate([km[:, 0], knm[:, 0]])

    @property
    def llrs_at95(self) -> np.ndarray:
        km = self.km_llr_data.llr_intervals
        knm = self.knm_llr_data.llr_intervals
        return np.concatenate([km[:, 1], knm[:, 1]])
