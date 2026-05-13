from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, HttpUrl

C = TypeVar("C", bound="URLContainer")


class URLContainer(BaseModel):
    @classmethod
    def from_enum(
        cls: type[C],
        enum: type[StrEnum],
        base_url: str,
    ) -> C:
        """Initiate the Response model with the given files from the enum."""
        return cls(**{file.name: HttpUrl(f"{base_url}/{file.value}") for file in enum})
