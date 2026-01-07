"""File service utilities for directory and resource management."""

from http import HTTPStatus
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from loguru import logger
from pydantic import HttpUrl

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


def get_files(resource_directory: Path, **filenames: str) -> dict[str, Path]:
    """
    Generate file paths within this directory's resource path.

    Creates a mapping of field names to absolute file paths by joining
    each filename with this directory's resource path.

    :param resource_directory: The resource directory path.
    :param filenames: Keyword arguments mapping field names to filenames.
                      Example: scan="scan.x3p", preview="preview.png"
    :return: Dictionary mapping field names to absolute Path objects.

    .. rubric:: Examples

    >>> vault = DirectoryAccess(tag="my-project")
    >>> paths = get_files(vault.resource_path, scan="scan.x3p", preview="preview.png")
    >>> paths["scan"]  # Returns Path like: /tmp/scratch_api/my-project-abc123/scan.x3p
    """
    return {field: resource_directory / file for field, file in filenames.items()}


def get_urls(access_url: str, **filenames: str) -> dict[str, HttpUrl]:
    """
    Generate access URLs for files within this directory.

    Creates a mapping of field names to HTTP URLs by joining each filename
    with this directory's access URL base path.

    :param access_url: The base access URL for the directory.
    :param filenames: Keyword arguments mapping field names to filenames.
                      Example: scan="scan.x3p", preview="preview.png"
    :return: Dictionary mapping field names to validated HttpUrl objects.

    .. rubric:: Examples

    >>> vault = DirectoryAccess(tag="my-project")
    >>> urls = get_urls(vault.access_url, scan="scan.x3p", preview="preview.png")
    >>> str(urls["scan"])  # Returns: http://localhost:8000/files/{token}/my-project/scan.x3p

    .. note::
       URLs are validated as proper HTTP URLs via Pydantic's HttpUrl type.
    """
    return {field: HttpUrl(url=f"{access_url}/{file}") for field, file in filenames.items()}
