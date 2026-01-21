from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, Field

from models import validate_file_extension, validate_relative_path


class SupportedExtension(StrEnum):
    X3P = auto()
    PNG = auto()


type RelativePath = Annotated[
    Path,
    AfterValidator(validate_relative_path),
    AfterValidator(lambda filename: validate_file_extension(filename, SupportedExtension)),
    Field(
        ...,
        description=f"Filename of type: {', '.join(SupportedExtension)}",
        examples=["example.png", "scan.x3p"],
    ),
]
