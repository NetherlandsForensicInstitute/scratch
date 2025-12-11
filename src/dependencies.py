from pathlib import Path

from fastapi import Request


def get_tmp_dir(request: Request) -> Path:
    """Get the temporary directory from the app state."""
    return Path(request.app.state.temp_dir.name)
