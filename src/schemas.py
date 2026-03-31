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


def generate_description(model: type[BaseModel]) -> str:
    """Generate a description field for in swagger docs for development purpose."""
    lines = ["\n\n---\n\nExpected form data:"]

    for name, field in model.model_fields.items():
        required = "required" if field.is_required() else "optional"
        default = f" (default: {field.default})" if field.default is not None and not field.is_required() else ""
        desc = field.description or ""

        lines.append(f"- `{name}` ({required}){default}\n  {desc}")

    return "\n".join(lines)
