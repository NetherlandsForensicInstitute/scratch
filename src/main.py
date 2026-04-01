import json
import shutil
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError
from uvicorn import run

from constants import LogLevel
from helpers import setup_logging
from preprocessors.exceptions import ArrayShapeMismatchError
from routers import prefix_router
from settings import get_settings

_PARSE_EXCEPTIONS = (json.JSONDecodeError, ValidationError, ValueError, KeyError)


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
    message = f"File not found error: {exc}"
    logger.warning(message)
    return JSONResponse(status_code=HTTPStatus.NOT_FOUND, content={"detail": message})


@app.exception_handler(ArrayShapeMismatchError)
async def array_shape_mismatch_handler(request: Request, exc: ArrayShapeMismatchError) -> JSONResponse:
    """Return a 422 JSON response for unhandled ArrayShapeMismatchError exceptions."""
    message = f"Array shape mismatch error: {exc}"
    logger.warning(message)
    return JSONResponse(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, content={"detail": message})


async def parse_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 422 JSON response for unhandled parse exceptions."""
    message = f"Data parsing error: {exc}"
    logger.warning(message)
    return JSONResponse(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, content={"detail": message})


for exc_type in _PARSE_EXCEPTIONS:
    app.add_exception_handler(exc_type, parse_exception_handler)


if __name__ == "__main__":
    settings = get_settings()
    logger.info("Starting server...")
    run(app, host=settings.api_host, port=settings.api_port, reload=False, workers=1)
