from pathlib import Path
from uuid import UUID

from constants import BASE_URL

from .router import ROUTE
from .schemas import ImageAccess


def get_file_access(temp_dir: Path, token: UUID) -> ImageAccess:
    """
    Generate a base URL for file retrieval and create the token directory.

    Creates a token-specific subdirectory within the temporary directory
    and returns the base URL that can be used to retrieve files from that directory.

    :param temp_dir: Temporary directory to store files.
    :param token: Unique token identifying the directory.
    :returns: Base URL string for file retrieval endpoints.
    """
    (resource_path := temp_dir / str(token)).mkdir(parents=True, exist_ok=True)
    return ImageAccess(resource_path, access_url=f"{BASE_URL}{ROUTE}/files/{token}")
