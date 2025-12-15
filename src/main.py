import tempfile
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from loguru import logger
from uvicorn import run

from comparators.router import comparison_router
from preprocessors import preprocessor_route
from processors.router import processors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create and clean up a temporary directory for the application lifespan.

    This context manager sets up a temporary directory when the FastAPI application
    starts and ensures it is cleaned up when the application shuts down.

    :param app: The FastAPI application instance.

    :yields: ``None``
    """
    temp_dir = tempfile.TemporaryDirectory(prefix="surface_comparator_")
    app.state.temp_dir = temp_dir
    logger.info(f"Temporary directory created at: {temp_dir.name}")
    try:
        yield
    finally:
        logger.info(f"Cleaning up temporary directory: {temp_dir.name}")
        temp_dir.cleanup()


app = FastAPI(lifespan=lifespan, title="Scratch API", version="0.1.0")
prefix_router = APIRouter()

prefix_router.include_router(preprocessor_route)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
app.include_router(prefix_router)


if __name__ == "__main__":
    logger.info("Starting server...")
    run(app, host="127.0.0.1", port=8000, reload=False, workers=1)
