from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )


def validate_file_extension(file_name: str, extensions: type[StrEnum]) -> str:
    """Validate that the file has a supported extension."""
    if not file_name.endswith(tuple(extensions)):
        raise ValueError(f"unsupported file type: {file_name}, try: {', '.join(extensions)}")
    return file_name
