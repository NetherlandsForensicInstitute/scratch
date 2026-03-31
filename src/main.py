import shutil
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from uvicorn import run

from constants import LogLevel
from helpers import setup_logging
from routers import prefix_router
from settings import get_settings


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """
    Initialize application resources on startup and cleanup on shutdown.

    This context manager configures the application with settings and
    manages the storage directory lifecycle.
    """
    setup_logging(LogLevel.INFO)  # TODO: We can move this config of loglevel to env
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


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Return a 404 JSON response for unhandled FileNotFoundError exceptions."""
    logger.warning(f"File not found: {exc}")
    return JSONResponse(status_code=HTTPStatus.NOT_FOUND, content={"detail": str(exc) or "File not found"})


if __name__ == "__main__":
    settings = get_settings()
    logger.info("Starting server...")
    run(app, host=settings.api_host, port=settings.api_port, reload=False, workers=1)
