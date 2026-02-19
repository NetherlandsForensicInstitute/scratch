from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from preprocessors.schemas import EditImage, MaskParameters, UploadScan

MASK = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
MASK_BYTES = MASK.tobytes(order="C")
MASK_SHAPE = MASK.shape
CUTOFF_LENGTH: Final[float] = 250


@pytest.fixture(scope="module")
def edit_image_parameter(scan_directory: Path) -> Callable[..., EditImage]:
    def wrapper(**kwargs) -> EditImage:
        return EditImage.model_validate(
            {
                "scan_file": scan_directory / "circle.x3p",
                "cutoff_length": CUTOFF_LENGTH,
                "mask_parameters": {"shape": MASK_SHAPE},
            }
            | kwargs
        )

    return wrapper


@pytest.fixture(scope="module")
def edit_image(scan_directory: Path) -> EditImage:
    return EditImage(
        scan_file=scan_directory / "circle.x3p",
        cutoff_length=CUTOFF_LENGTH,
        mask_parameters=MaskParameters(shape=MASK_SHAPE),  # type: ignore
    )  # type: ignore


@pytest.fixture(scope="module")
def upload_scan_parameter(scan_directory: Path) -> Callable[..., UploadScan]:
    def wrapper(**kwargs) -> UploadScan:
        return UploadScan.model_validate({"scan_file": scan_directory / "circle.x3p"} | kwargs)

    return wrapper


@pytest.fixture(scope="module")
def upload_scan(scan_directory: Path) -> UploadScan:
    return UploadScan(scan_file=scan_directory / "circle.x3p")  # type: ignore
