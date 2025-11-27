from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from uvicorn import run

from comparators.router import comparison_router
from preprocessors import preprocessor_route
from preprocessors.exceptions import ScanImageException
from processors.router import processors

app = FastAPI()
prefix_router = APIRouter()

prefix_router.include_router(preprocessor_route)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
app.include_router(prefix_router)


@app.exception_handler(ScanImageException)
async def scan_image_exception_handler(request: Request, exc: ScanImageException):
    """Exception handler for ScanImageException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message},
    )


if __name__ == "__main__":
    logger.info("Starting server...")
    run(app, host="127.0.0.1", port=8000, reload=False, workers=1)
