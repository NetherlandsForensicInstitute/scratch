from pydantic import BaseModel, ConfigDict


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        regex_engine="rust-regex",
        extra="forbid",
    )
