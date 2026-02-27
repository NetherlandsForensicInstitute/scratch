from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, ClassVar, Generic, Protocol, Self, Type, TypeVar, cast

from pydantic import AfterValidator, BaseModel, Field, HttpUrl, create_model, model_serializer

from extractors.constants import (
    ComparisonImpressionFiles,
    ComparisonStriationFiles,
    GeneratedImageFiles,
    LRFiles,
    PrepareMarkImpressionFiles,
    PrepareMarkStriationFiles,
    ProcessFiles,
    UrlFiles,
)
from models import (
    BaseModelConfig,
    validate_file_extension,
    validate_relative_path,
)


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


F = TypeVar("F", bound=StrEnum)
C = TypeVar("C", bound="URLContainer")


class URLContainer(BaseModel, Generic[F]):
    @classmethod
    def from_enum(
        cls: type[C],
        enum: type[F],
        base_url: str,
    ) -> C:
        return cls(**{file.name: f"{base_url}/{file.value}" for file in cls._files})


def response_model_from_enum(name: str, files: type[StrEnum]) -> type[URLContainer]:
    """Generate a Response model for the FastAPI server."""
    return create_model(name, **{file.name: (HttpUrl, None) for file in files})  # type: ignore


def generate_model_with_urls(
    name: str,
    files: type[StrEnum],
    base_url: str,
) -> URLContainer:
    """Generate pydantic response model with urls initiated."""
    data = {file.name: f"{base_url}/{file.value}" for file in files}
    model = create_model(
        name,
        **{file.name: (HttpUrl, None) for file in files},  # type: ignore
    )
    return model.model_validate(data)


ComparisonResponseStriation = response_model_from_enum(
    "ComparisonResponseStriation",
    ComparisonImpressionFiles,
)
ComparisonResponseImpression = response_model_from_enum(
    "ComparisonResponseImpression",
    ComparisonStriationFiles,
)
PrepareMarkResponseImpression = response_model_from_enum(
    "PrepareMarkResponseImpression",
    PrepareMarkImpressionFiles,
)
PrepareMarkResponseStriation = response_model_from_enum(
    "PrepareMarkResponseStriation",
    PrepareMarkStriationFiles,
)
ProcessedDataAccess = response_model_from_enum(
    "ProcessedDataAccess",
    ProcessFiles,
)

LRResponseURL = response_model_from_enum(
    "LRResponseURL",
    LRFiles,
)


class GeneratedImages(URLContainer[GeneratedImageFiles]):
    preview_image: HttpUrl
    surface_map_image: HttpUrl


class LRResponse(BaseModel):
    urls: LRResponseURL
    lr: float

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        """Serialize model to flat json."""
        data = handler(self)
        return {
            **data["urls"],
            "lr": data["lr"],
        }
