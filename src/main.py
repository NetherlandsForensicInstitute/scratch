import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from uvicorn import run

from routers import prefix_router
from settings import get_settings


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """
    Initialize application resources on startup and cleanup on shutdown.

    This context manager configures the application with settings and
    manages the storage directory lifecycle.
    """
    settings = get_settings()
    settings.log_startup_config()

    yield  # Application runs here

    # Cleanup storage directory on shutdown if storage is NOT persistent
    if not settings.persistent_storage:
        try:
            if settings.storage.exists():
                shutil.rmtree(settings.storage)
                logger.info(f"Cleaned up default storage directory: {settings.storage}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to cleanup storage directory {settings.storage}: {e}")
    elif settings.persistent_storage:
        logger.info(f"Persistent storage enabled - keeping directory: {settings.storage}")

    logger.info("Application shutdown")


settings = get_settings()
app = FastAPI(
    lifespan=_lifespan,
    title=settings.app_title,
    version=settings.app_version,
    redirect_slashes=False,
)
app.include_router(prefix_router)


if __name__ == "__main__":
    settings = get_settings()
    logger.info("Starting server...")
    run(app, host=settings.api_host, port=settings.api_port, reload=False, workers=1)
