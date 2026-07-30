import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from constants import LogLevel


def setup_logging(level: LogLevel = LogLevel.INFO):
    """
    Configure Loguru logging for the backend.

    Adds a file handler in the project root directory with:
    - Retention of the last 5 log files
    - Rotation at 00:00, new is made file every day.
    - File is saved in the root of the project.

    :param level: Log level to use (e.g. "DEBUG", "INFO", "WARNING"), Defaulting to "INFO".
    :return: None
    """
    project_root = Path(__file__).resolve().parent
    log_file = project_root / f"backend-{datetime.date.today()}.log"
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="00:00",
        retention=5,
        compression="zip",
        encoding="utf-8",
    )
    logger.debug("Logging initialized")


def generate_openapi_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Generate the OpenAPI schema for multipart/form-data endpoints.

    Swagger UI cannot resolve $ref pointers when they're nested inside multipart form-data schemas.
    This function inlines all references so the docs render correctly.
    """
    model_schema = model.model_json_schema()
    defs = model_schema.pop("$defs", {})
    params_schema = _inline_refs(model_schema, defs)

    return {
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "properties": {
                            "params": params_schema,
                            "mask_data": {"type": "string", "format": "binary", "example": b"\x01\x00\x00\x01"},
                        },
                        "required": ["params", "mask_data"],
                    }
                },
                "application/json": {
                    "schema": {
                        "properties": {"params": params_schema},
                        "required": ["params"],
                    }
                },
            }
        }
    }


def _resolve_ref(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any] | None:
    """
    Resolve a single $ref to its definition, merging any sibling properties.

    Returns None if the ref cannot be resolved (not a $ref, or missing from defs).
    """
    if "$ref" not in schema:
        return None

    ref_key = schema["$ref"].removeprefix("#/$defs/")
    if ref_key not in defs:
        return None

    # Start with the resolved definition (recursively inlined)
    resolved = _inline_refs(defs[ref_key], defs)
    # Merge any sibling properties (e.g. description, default) on top
    for key, value in schema.items():
        if key != "$ref":
            resolved[key] = _inline_refs(value, defs)
    return resolved


def _inline_refs(schema: Any, defs: dict[str, Any]) -> Any:
    """
    Recursively inline all $ref references using the provided $defs.

    This is needed for multipart/form-data schemas where $defs are nested inside the params property
    and JSON Pointer $ref resolution fails because refs resolve relative to the OpenAPI document root,
    not the nested schema location.
    """
    if isinstance(schema, dict):
        # Try to resolve $ref first (takes priority over recursing into keys)
        resolved = _resolve_ref(schema, defs)
        if resolved is not None:
            return resolved
        return {key: _inline_refs(value, defs) for key, value in schema.items()}
    if isinstance(schema, list):
        return [_inline_refs(item, defs) for item in schema]

    return schema
