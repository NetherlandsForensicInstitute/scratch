"""Application settings and configuration."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from loguru import logger
from pydantic import DirectoryPath, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _create_default_temp_dir() -> Path:
    """
    Create and return the default temporary directory.

    :return: The created temporary directory path.
    :raises:
        OSError: If directory creation fails.
        PermissionError: If the application lacks permissions to create the directory.
    """
    temp_dir = Path(tempfile.gettempdir()) / "scratch_api"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Default storage directory created: {temp_dir}")
        return temp_dir
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create default storage directory {temp_dir}: {e}")
        raise


class Settings(BaseSettings):
    """
    Application configuration settings.

    Settings can be configured via:

    1. Environment variables (e.g., SCRATCH_TEMP_DIR=/custom/path)
    2. .env file in the project root
    3. Default values defined below

    All settings use the SCRATCH_ prefix for environment variables.

    .. rubric:: Examples

    Set temp directory via environment::

        export SCRATCH_TEMP_DIR=/custom/temp
        export SCRATCH_MAX_UPLOAD_SIZE=52428800  # 50MB

    Or create a .env file::

        SCRATCH_TEMP_DIR=/custom/temp
        SCRATCH_MAX_UPLOAD_SIZE=52428800
    """

    # Storage Configuration
    storage: Annotated[
        DirectoryPath,
        Field(
            default_factory=_create_default_temp_dir,
            alias="STORAGE",
            description="Base directory for file storage",
        ),
    ]

    persistent_storage: Annotated[
        bool,
        Field(
            default=False,
            alias="PERSISTENT_STORAGE",
            description="If True, files in storage persist between application restarts. "
            "If False, storage directory is cleaned up on shutdown.",
        ),
    ]

    max_upload_size: Annotated[
        int,
        Field(
            default=100 * 1024 * 1024,  # 100MB
            alias="MAX_UPLOAD_SIZE",
            description="Maximum upload file size in bytes",
            gt=0,
        ),
    ]

    # API Configuration
    api_host: Annotated[str, Field(default="127.0.0.1", alias="API_HOST", description="API host address")]
    api_port: Annotated[int, Field(default=8000, alias="API_PORT", description="API port", gt=0, lt=65536)]

    # Application Metadata
    app_title: Annotated[str, Field(default="Scratch API", alias="APP_TITLE", description="Application title")]

    model_config = SettingsConfigDict(
        env_prefix="SCRATCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        populate_by_name=True,
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def app_version(self) -> str:
        """
        Get the application version from package metadata.

        :return: The application version from pyproject.toml.
                 Falls back to "0.0.0" if version cannot be determined.
        """
        try:
            return version("scratch")
        except Exception:
            logger.warning("Could not determine package version, using fallback '0.0.0'")
            return "0.0.0"

    def log_startup_config(self) -> None:
        """
        Log application configuration at startup.

        Logs all settings in a readable format with special formatting
        for certain fields like file sizes and combined host/port.
        """
        logger.info("=" * 60)
        logger.info("Application startup - Configuration:")
        logger.info(f"  Title: {self.app_title}")
        logger.info(f"  Version: {self.app_version}")
        logger.info(f"  Host: {self.api_host}:{self.api_port}")
        logger.info(f"  Storage: {self.storage}")
        logger.info(f"  Persistent storage: {self.persistent_storage}")
        logger.info(f"  Max upload size: {self.max_upload_size / 1024 / 1024:.1f}MB")
        logger.info("=" * 60)

    @property
    def base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function is cached to ensure only one Settings instance
    is created per application lifecycle, following FastAPI best practices.

    :return: The application settings instance.
    """
    return Settings()  # type: ignore


# Type alias for cleaner endpoint signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
