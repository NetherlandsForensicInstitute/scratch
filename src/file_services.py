"""File service utilities for directory and resource management."""

from http import HTTPStatus
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from loguru import logger

from models import DirectoryAccess
from settings import get_settings


def create_vault(tag: str) -> DirectoryAccess:
    """
    Create a new DirectoryAccess instance and initialize its storage directory.

    This factory method creates a DirectoryAccess instance with a unique token
    and ensures the corresponding directory exists on the filesystem.

    :param tag: The tag identifier for the directory (e.g., project name or scan name).
    :return: A new DirectoryAccess instance with an initialized storage directory.
    :raises HTTPException: If directory creation fails due to OS or permission errors.
    """
    vault = DirectoryAccess(tag=tag)
    try:
        vault.resource_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create directory {vault}: {e}")
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "Unable to allocate space for resources")
    return vault


def fetch_resource_path(token: UUID) -> Path:
    """
    Fetch the resource directory path by verifying filesystem existence.

    Validates that a directory exists for the given token and tag combination,
    then returns the path to that directory.

    :param token: The unique token identifying the directory.
    :param tag: The tag associated with the directory.
    :return: The filesystem path to the resource directory.
    :raises HTTPException: 422 status code if directory does not exist for the given token and tag.

    .. note::
       The directory must exist on the filesystem before calling this function.
       The path is constructed using the pattern: {storage}/{tag}-{token.hex}
    """
    if resource := next(get_settings().storage.glob(f"*-{token.hex}", case_sensitive=True), None):
        return resource
    logger.error(f"Directory not found for token '{token}'")
    raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, f"Unable to fetch resources of token '{token}'")
