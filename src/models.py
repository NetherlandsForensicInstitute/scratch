from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )


def validate_file_extension(filename: str, extensions: type[StrEnum]) -> str:
    """Validate that the file has a supported extension."""
    if not filename.endswith(tuple(extensions)):
        raise ValueError(f"unsupported file type: {filename}, try: {', '.join(extensions)}")
    return filename
