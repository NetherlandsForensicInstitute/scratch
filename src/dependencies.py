from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Request


def get_tmp_dir(request: Request) -> Path:
    """Get the temporary directory from the app state."""
    return Path(request.app.state.temp_dir.name)


def get_token() -> UUID:
    """Generate a unique token for file storage."""
    return uuid4()
