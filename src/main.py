from fastapi import APIRouter, FastAPI
from loguru import logger
from uvicorn import run

from add_scan.router import add_scan
from comparators.router import comparison_router
from preprocessors import preprocessor_route
from processors.router import processors

app = FastAPI()
prefix_router = APIRouter()

prefix_router.include_router(preprocessor_route)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
app.include_router(prefix_router)


if __name__ == "__main__":
    logger.info("Starting server...")
    run(app, host="127.0.0.1", port=8000, reload=False, workers=1)
